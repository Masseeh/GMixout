# Robust and Efficient Finetuning of Vision Foundation Models via Implicit Ensembling (GMixout)

This is the source code for the paper: [Robust and Efficient Finetuning of Vision Foundation Models via Implicit Ensembling](https://arxiv.org/abs/2510.06982) (Neurips 2026).

## Requirements

* Python 3.10

```bash
virtualenv --no-download .venv
source .venv/bin/activate
pip install --no-index -r requirements.txt
```


### Prepare the Dataset

Put files in the following locations and change the path in the data configure files in [configs/data](configs/data), e.g:

- ImageNet

```
Path/To/Dataset
├─ train
│  ├─ n01440764
|  |  ├─ n01440764_18.JPEG
|  |  └─ ......
│  └─ ......
└─ val
   ├─ n01440764
   |  ├─ ILSVRC2012_val_00000293.JPEG
   |  └─ ......
   └─ ......
```

### Reproduction

Use [scripts/slurm_run.sh](scripts/slurm_run.sh) to run the experiments, e.g:

#### DomainNet

```bash
for domains in domainnet_r domainnet_p domainnet_s domainnet_c
do 

bash scripts/slurm_run.sh --dataset $domains --name gmixout --ft maskr --epochs 10 head_tuning False mask_ema 0.5 mask_refresh 200

bash scripts/slurm_run.sh --dataset ${domains}_shift --name gmixout --ft maskr --eval_conf default --model_dir output/${domains}_clip_gmixout/checkpoint.pth.tar

done
```

#### iWildCam

```bash

bash scripts/slurm_run.sh --dataset iwildcam_id --name gmixout --ft maskr --epochs 20 head_tuning True mask_ema 0.3 mask_refresh 20 early_stop 5 lr 1e-4

bash scripts/slurm_run.sh --dataset iwildcam_ood --name gmixout --ft maskr --eval_conf default --model_dir output/iwildcam_id_clip_gmixout/checkpoint.pth.tar

done
```

#### ImageNet

```bash

bash scripts/slurm_run.sh --dataset imagenet --name gmixout --ft maskr --epochs 10 mask_ema 0.5 mask_refresh 50

bash scripts/slurm_run.sh --dataset imagenet_shift --name gmixout --ft maskr --eval_conf default --model_dir output/imagenet_clip_gmixout/checkpoint.pth.tar

```
