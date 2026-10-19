from typing import Optional

import torch


class RankAllocator:
    """
    Rank allocator for AdaLoRA modules with:

        lora_A: (in_dim, rank)
        lora_E: (1, rank)
        lora_B: (rank, out_dim)
    """

    def __init__(
        self,
        model,
        lora_r: int,
        target_rank: int,
        init_warmup: int,
        final_warmup: int,
        mask_interval: int,
        beta1: float,
        beta2: float,
        total_step: Optional[int] = None,
        target_total_rank: Optional[int] = None,
        tb_writer=None,
        tb_writer_loginterval: int = 500,
    ):
        if not 0.0 < beta1 < 1.0:
            raise ValueError("beta1 must be between 0 and 1.")

        if not 0.0 < beta2 < 1.0:
            raise ValueError("beta2 must be between 0 and 1.")

        self.ave_target_rank = target_rank
        self.target_rank = target_total_rank
        self.lora_init_rank = lora_r

        self.initial_warmup = init_warmup
        self.final_warmup = final_warmup
        self.mask_interval = mask_interval

        self.beta1 = beta1
        self.beta2 = beta2
        self.total_step = total_step
        self.model = model

        self.ipt = {}
        self.exp_avg_ipt = {}
        self.exp_avg_unc = {}

        self.rank_pattern = {}

        self.tb_writer = tb_writer
        self.log_interval = tb_writer_loginterval

        self.global_step = 0

        self.get_lora_param_name()

        if self.total_step is not None:
            self._validate_total_step()

    def _validate_total_step(self):
        if self.total_step <= (
            self.initial_warmup + self.final_warmup
        ):
            raise ValueError(
                "total_step must be larger than "
                "init_warmup + final_warmup."
            )

    def set_total_step(self, total_step: int):
        self.total_step = total_step
        self._validate_total_step()

    def get_rank_pattern(self):
        return self.rank_pattern

    def get_lora_param_name(self):
        """
        Discover AdaLoRA triplets and calculate the initial total rank.

        In this implementation, rank is:
            A.shape[1]
            E.shape[1]
            B.shape[0]
        """
        self.name_set = set()
        self.total_rank = 0
        self.shape_dict = {}

        for name, parameter in self.model.named_parameters():
            if "lora_A" in name:
                name_pattern = name.replace("lora_A", "{}")
                self.name_set.add(name_pattern)

                # A is (in_dim, rank).
                self.total_rank += parameter.shape[1]
                self.shape_dict[name] = parameter.shape

            elif "lora_B" in name:
                self.shape_dict[name] = parameter.shape

            elif "lora_E" in name:
                self.shape_dict[name] = parameter.shape

        self.name_set = sorted(self.name_set)

        if self.target_rank is None:
            self.target_rank = (
                self.ave_target_rank * len(self.name_set)
            )

        if self.target_rank > self.total_rank:
            raise ValueError(
                f"Target total rank {self.target_rank} exceeds "
                f"initial total rank {self.total_rank}."
            )

    def schedule_threshold(self, step: int):
        if self.total_step is None:
            raise RuntimeError(
                "total_step is not set. Pass it to __init__() or "
                "call set_total_step()."
            )

        self.global_step = step

        if step <= self.initial_warmup:
            # Keep every rank component during initial warmup.
            curr_rank = self.total_rank
            mask_ind = False

        elif step > self.total_step - self.final_warmup:
            # Keep the final rank pattern fixed.
            curr_rank = self.target_rank
            mask_ind = True

        else:
            # Cubic rank-budget decay used by AdaLoRA.
            denominator = (
                self.total_step
                - self.final_warmup
                - self.initial_warmup
            )

            progress = (
                step - self.initial_warmup
            ) / denominator

            multiplier = 1.0 - progress

            curr_rank = (
                self.target_rank
                + (self.total_rank - self.target_rank)
                * multiplier**3
            )
            curr_rank = int(curr_rank)

            mask_ind = step % self.mask_interval == 0

        return curr_rank, mask_ind

    def update_ipt(self, model):
        """
        Update element-wise importance and uncertainty estimates.

        This must be called after loss.backward(), because p.grad is needed.
        """
        for name, parameter in model.named_parameters():
            if not any(
                token in name
                for token in ("lora_A", "lora_B", "lora_E")
            ):
                continue

            if parameter.grad is None:
                continue

            if name not in self.ipt:
                self.ipt[name] = torch.zeros_like(parameter)
                self.exp_avg_ipt[name] = torch.zeros_like(parameter)
                self.exp_avg_unc[name] = torch.zeros_like(parameter)

            with torch.no_grad():
                importance = (
                    parameter * parameter.grad
                ).abs().detach()

                self.ipt[name] = importance

                self.exp_avg_ipt[name].mul_(self.beta1).add_(
                    importance,
                    alpha=1.0 - self.beta1,
                )

                deviation = (
                    importance - self.exp_avg_ipt[name]
                ).abs()

                self.exp_avg_unc[name].mul_(self.beta2).add_(
                    deviation,
                    alpha=1.0 - self.beta2,
                )

    def calculate_score(
        self,
        name,
        parameter=None,
        metric="ipt",
    ):
        if metric == "ipt":
            return (
                self.exp_avg_ipt[name]
                * self.exp_avg_unc[name]
            )

        if metric == "mag":
            if parameter is None:
                raise ValueError(
                    "parameter is required for metric='mag'."
                )

            return parameter.detach().abs().clone()

        raise ValueError(f"Unexpected metric: {metric}")

    @staticmethod
    def _combine_ipt(ipt_E, ipt_AB):
        """
        All inputs represent one score per rank component.

        ipt_E:  (1, rank) or flattened equivalent
        ipt_AB: (rank, 2), containing A and B scores
        """
        ab_score = ipt_AB.sum(dim=1)

        return ipt_E.reshape(-1) + ab_score.reshape(-1)

    def mask_to_target_rank(self, model, curr_rank):
        importance_by_e = {}
        factor_scores = {}
        singular_scores = {}

        # Calculate one importance score per rank component.
        for name, parameter in model.named_parameters():
            if name not in self.exp_avg_ipt:
                continue

            name_pattern = None

            if "lora_A" in name:
                # A is (in_dim, rank).
                # Average over in_dim, preserving rank.
                score = self.calculate_score(name, metric="ipt")
                rank_score = score.mean(dim=0).reshape(-1, 1)

                name_pattern = name.replace("lora_A", "{}")

            elif "lora_B" in name:
                # B is (rank, out_dim).
                # Average over out_dim, preserving rank.
                score = self.calculate_score(name, metric="ipt")
                rank_score = score.mean(dim=1).reshape(-1, 1)

                name_pattern = name.replace("lora_B", "{}")

            elif "lora_E" in name:
                score = self.calculate_score(
                    name,
                    parameter=parameter,
                    metric="ipt",
                )

                name_pattern = name.replace("lora_E", "{}")
                singular_scores[name_pattern] = score
                continue

            else:
                continue

            factor_scores.setdefault(
                name_pattern,
                [],
            ).append(rank_score)

        all_scores = []

        for name_pattern, scores in factor_scores.items():
            if name_pattern not in singular_scores:
                raise RuntimeError(
                    "Found lora_A/lora_B without matching lora_E: "
                    f"{name_pattern}"
                )

            if len(scores) != 2:
                raise RuntimeError(
                    "Expected one lora_A and one lora_B for "
                    f"{name_pattern}, found {len(scores)}."
                )

            ipt_E = singular_scores[name_pattern]
            ipt_AB = torch.cat(scores, dim=1)

            combined_score = self._combine_ipt(
                ipt_E=ipt_E,
                ipt_AB=ipt_AB,
            )

            e_name = name_pattern.format("lora_E")

            # E has shape (1, rank), so the mask must also have
            # shape (1, rank).
            importance_by_e[e_name] = combined_score.reshape(1, -1)
            all_scores.append(combined_score)

        if not all_scores:
            raise RuntimeError(
                "No complete AdaLoRA parameter groups were found."
            )

        global_scores = torch.cat(all_scores)

        num_to_prune = self.total_rank - curr_rank

        if num_to_prune <= 0:
            return None

        if num_to_prune >= global_scores.numel():
            raise ValueError(
                "The requested rank budget would prune every "
                "rank component."
            )

        threshold = torch.kthvalue(
            global_scores,
            k=num_to_prune,
        ).values.item()

        with torch.no_grad():
            current_total_rank = 0
            active_parameter_count = 0

            for name, parameter in model.named_parameters():
                if "lora_E" not in name:
                    continue

                score = importance_by_e.get(name)

                if score is None:
                    continue

                active_mask = score > threshold

                parameter.masked_fill_(
                    ~active_mask,
                    0.0,
                )

                ranknum = int(active_mask.sum().item())
                self.rank_pattern[name] = ranknum

                current_total_rank += ranknum

                a_shape = self.shape_dict[
                    name.replace("lora_E", "lora_A")
                ]
                b_shape = self.shape_dict[
                    name.replace("lora_E", "lora_B")
                ]

                # Each active rank component contributes:
                # one A column of length in_dim
                # one B row of length out_dim
                active_parameter_count += ranknum * a_shape[0]
                active_parameter_count += ranknum * b_shape[1]

                # Keep ranknum synchronized with the number of
                # active singular values.
                ranknum_name = name.replace(
                    "lora_E",
                    "ranknum",
                )

                ranknum_parameter = dict(
                    model.named_parameters()
                ).get(ranknum_name)

                if ranknum_parameter is not None:
                    ranknum_parameter.fill_(float(ranknum))

                if (
                    self.tb_writer is not None
                    and self.global_step % self.log_interval == 0
                ):
                    self.tb_writer.add_scalar(
                        f"Ranknum/{name}",
                        ranknum,
                        self.global_step,
                    )

            if (
                self.tb_writer is not None
                and self.global_step % self.log_interval == 0
            ):
                self.tb_writer.add_scalar(
                    "Budget/total_rank",
                    current_total_rank,
                    self.global_step,
                )
                self.tb_writer.add_scalar(
                    "Budget/mask_threshold",
                    threshold,
                    self.global_step,
                )
                self.tb_writer.add_scalar(
                    "Budget/sum_param",
                    active_parameter_count,
                    self.global_step,
                )

        return threshold

    def update_and_mask(self, model, global_step):
        if global_step < (
            self.total_step - self.final_warmup
        ):
            self.update_ipt(model)

        curr_rank, mask_ind = self.schedule_threshold(global_step)

        if mask_ind:
            threshold = self.mask_to_target_rank(
                model=model,
                curr_rank=curr_rank,
            )
        else:
            threshold = None

        self._maybe_tb_writer_log(model)

        return curr_rank, threshold

    def _maybe_tb_writer_log(self, model):
        if self.tb_writer is None:
            return

        if self.global_step % self.log_interval != 0:
            return

        losses = []

        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if "lora_A" in name:
                    covariance = parameter.T @ parameter
                elif "lora_B" in name:
                    covariance = parameter @ parameter.T
                else:
                    continue

                identity = torch.eye(
                    covariance.shape[0],
                    device=covariance.device,
                    dtype=covariance.dtype,
                )

                orth_loss = torch.linalg.matrix_norm(
                    covariance - identity,
                    ord="fro",
                )

                losses.append(orth_loss.item())

                self.tb_writer.add_scalar(
                    f"Orth_regu_loss/{name}",
                    orth_loss.item(),
                    self.global_step,
                )

        if losses:
            self.tb_writer.add_scalar(
                "train/orth_regu_loss",
                sum(losses) / len(losses),
                self.global_step,
            )

def compute_orth_regu(model, regu_weight=0.1):
    """
    Orthogonality regularization for:

        lora_A: (in_dim, rank)
        lora_B: (rank, out_dim)

    Encourages:
        A.T @ A ≈ I_rank
        B @ B.T ≈ I_rank
    """
    losses = []

    for name, parameter in model.named_parameters():
        if "lora_A" in name:
            covariance = parameter.T @ parameter

        elif "lora_B" in name:
            covariance = parameter @ parameter.T

        else:
            continue

        identity = torch.eye(
            covariance.shape[0],
            device=covariance.device,
            dtype=covariance.dtype,
        )

        losses.append(
            torch.linalg.matrix_norm(
                covariance - identity,
                ord="fro",
            )
        )

    if not losses:
        # Return a differentiable zero on the model's device.
        first_parameter = next(model.parameters())

        return first_parameter.sum() * 0.0

    return regu_weight * torch.stack(losses).mean()