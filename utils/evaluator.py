import numpy as np
import torch.distributed as dist
from collections import OrderedDict, defaultdict
import torch
from sklearn.metrics import f1_score
import logging

class Evaluator:
    """Evaluator for classification."""

    def __init__(self, cfg, cls_num_list, eval_fn=None, stout=True):
        self.cfg = cfg
        self.cls_num_list = cls_num_list
        self.reset()
        self.stout = stout
        self.eval_fn = eval_fn

        if cfg.dataset.endswith("_LT") or cfg.dataset == "iNaturalist2018":
            self.many_idxs = (np.array(cls_num_list) > 100).nonzero()[0]
            self.med_idxs = ((np.array(cls_num_list) >= 20) & (np.array(cls_num_list) <= 100)).nonzero()[0]
            self.few_idxs = (np.array(cls_num_list) < 20).nonzero()[0]
        else:
            self.many_idxs, self.med_idxs, self.few_idxs = None, None, None

    def reset(self):
        self._correct = 0
        self._total = 0
        self._y_true = []
        self._y_pred = []
        self._y_conf = []  # Store prediction confidences
        self._loss = []

    def process(self, mo, gt):
        pred = mo.max(1)[1]
        conf = torch.softmax(mo, dim=1).max(1)[0]
        matches = pred.eq(gt).float()
        self._correct += int(matches.sum().item())
        self._total += gt.shape[0]

        self._y_true.extend(gt.float().cpu().numpy().tolist())
        self._y_pred.extend(pred.float().cpu().numpy().tolist())
        self._y_conf.extend(conf.float().cpu().numpy().tolist())

        np_loss = torch.nn.functional.cross_entropy(mo, gt, reduction='none').float().cpu().numpy()
        self._loss.extend(np_loss.tolist())

    @torch.no_grad()
    def all_gather(self):
        y_true_tensor = torch.tensor(self._y_true, dtype=torch.float32, device='cuda')
        self._y_true = self._all_gather_tensor(y_true_tensor)

        y_pred_tensor = torch.tensor(self._y_pred, dtype=torch.float32, device='cuda')
        self._y_pred = self._all_gather_tensor(y_pred_tensor)

        y_conf_tensor = torch.tensor(self._y_conf, dtype=torch.float32, device='cuda')
        self._y_conf = self._all_gather_tensor(y_conf_tensor)

        loss_tensor = torch.tensor(self._loss, dtype=torch.float32, device='cuda')
        self._loss = self._all_gather_tensor(loss_tensor)

        total_tensor = torch.tensor([self._correct, self._total], dtype=torch.int64, device='cuda')
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
        self._correct, self._total = total_tensor.tolist()
        
    def _all_gather_tensor(self, tensor):
        """Gather a tensor from all processes."""
        world_size = dist.get_world_size()
        gather_list = [torch.empty_like(tensor) for _ in range(world_size)]
        dist.all_gather(gather_list, tensor)
        return torch.cat(gather_list, dim=0).cpu().numpy().tolist()

    def evaluate(self):
        results = OrderedDict()
        uacc = 100.0 * self._correct / self._total
        err = 100.0 - uacc
        macro_f1 = 100.0 * f1_score(
            self._y_true,
            self._y_pred,
            average="macro",
            labels=np.unique(self._y_true)
        )

        self._loss = np.array(self._loss)

        # The first value will be returned by trainer.test()
        results["loss"] = self._loss.mean()
        results["acc"] = uacc
        results["error_rate"] = err
        results["macro_f1"] = macro_f1

        per_class_res = defaultdict(list)
        per_class_loss = defaultdict(list)

        for label, pred, l in zip(self._y_true, self._y_pred, self._loss):
            matches = int(label == pred)
            per_class_res[label].append(matches)
            per_class_loss[label].append(l)

        labels = list(per_class_res.keys())
        labels.sort()

        cls_accs = []
        cls_losses = []
        for label in labels:
            res = per_class_res[label]
            correct = sum(res)
            total = len(res)
            acc = 100.0 * correct / total
            cls_accs.append(acc)
            cls_losses.append(np.mean(per_class_loss[label]))
        
        accs_string = np.array2string(np.array(cls_accs), precision=2)

        if self.stout:
            logging.eval(f"* class acc: {accs_string}")

        # Compute worst case accuracy
        worst_case_acc = min([acc for acc in cls_accs])

        # Compute harmonic mean
        hmean_acc = 100.0 / np.mean([1.0 / (max(acc, 0.001) / 100.0) for acc in cls_accs])

        # Compute geometric mean
        gmean_acc = 100.0 * np.prod([acc / 100.0 for acc in cls_accs]) ** (1.0 / len(cls_accs))

        # balanced accuracy
        bal_acc = np.mean(np.array(cls_accs))

        # balanced loss
        bal_loss = np.mean(np.array(cls_losses))

        lt_log = ""
        if self.many_idxs is not None and self.med_idxs is not None and self.few_idxs is not None:
            many_acc = np.mean(np.array(cls_accs)[self.many_idxs])
            med_acc = np.mean(np.array(cls_accs)[self.med_idxs])
            few_acc = np.mean(np.array(cls_accs)[self.few_idxs])
            results["many_acc"] = many_acc
            results["med_acc"] = med_acc
            results["few_acc"] = few_acc

            lt_log = f"\n* many: {many_acc:.1f}%  med: {med_acc:.1f}%  few: {few_acc:.1f}%"

        results["worst_case_acc"] = worst_case_acc
        results["hmean_acc"] = hmean_acc
        results["gmean_acc"] = gmean_acc
        results["bal_acc"] = bal_acc
        results["bal_loss"] = bal_loss

        dataset_specific_eval = self.eval_fn(torch.tensor(self._y_pred), torch.tensor(self._y_true))[1] if self.eval_fn is not None else None
        ds_log = f"\nDataset specific eval:\n{dataset_specific_eval}" if dataset_specific_eval is not None else ""

        if self.stout:
            logging.eval(
                f"* total: {self._total:,}\n"
                f"* correct: {self._correct:,}\n"
                f"* accuracy: {uacc:.1f}%\n"
                f"* error: {err:.1f}%\n"
                f"* macro_f1: {macro_f1:.1f}%\n"
                f"* worst_case_acc: {worst_case_acc:.1f}%\n"
                f"* hmean_acc: {hmean_acc:.1f}%\n"
                f"* gmean_acc: {gmean_acc:.1f}%\n"
                f"* bal_acc: {bal_acc:.1f}%\n"
                f"* bal_loss: {bal_loss:.4f}"
                f"{lt_log}"
                f"{ds_log}"
            )

        return results


def compute_accuracy(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for
    the specified values of k.

    Args:
        output (torch.Tensor): prediction matrix with shape (batch_size, num_classes).
        target (torch.LongTensor): ground truth labels with shape (batch_size).
        topk (tuple, optional): accuracy at top-k will be computed. For example,
            topk=(1, 5) means accuracy at top-1 and top-5 will be computed.

    Returns:
        list: accuracy at top-k.
    """
    maxk = max(topk)
    batch_size = target.size(0)

    if isinstance(output, (tuple, list)):
        output = output[0]

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        acc = correct_k.mul_(100.0 / batch_size)
        res.append(acc)

    return res


def expected_calibration_error(confs, preds, labels, num_bins=10):
    def _populate_bins(confs, preds, labels, num_bins):
        bin_dict = defaultdict(lambda: {'bin_accuracy': 0, 'bin_confidence': 0, 'count': 0})
        bins = np.linspace(0, 1, num_bins + 1)
        for conf, pred, label in zip(confs, preds, labels):
            bin_idx = np.searchsorted(bins, conf) - 1
            bin_dict[bin_idx]['bin_accuracy'] += int(pred == label)
            bin_dict[bin_idx]['bin_confidence'] += conf
            bin_dict[bin_idx]['count'] += 1
        return bin_dict

    bin_dict = _populate_bins(confs, preds, labels, num_bins)
    num_samples = len(labels)
    ece = 0
    for i in range(num_bins):
        bin_accuracy = bin_dict[i]['bin_accuracy']
        bin_confidence = bin_dict[i]['bin_confidence']
        bin_count = bin_dict[i]['count']
        ece += (float(bin_count) / num_samples) * \
               abs(bin_accuracy / bin_count - bin_confidence / bin_count)
    return ece
