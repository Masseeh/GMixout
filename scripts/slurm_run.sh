#!/bin/bash
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:v100l:4
#SBATCH --time=7:0:0    
#SBATCH --output=./slurm_logs/R-%x.%j.out
#SBATCH --mail-user=<masseeh.amin@gmail.com>
#SBATCH --mail-type=ALL

# abort on error
set -e

ft=full # full/lora/mask/maskr
seed=0
ema=0.0
test_batch_size=256
epochs=10
cluster_bs=250
o3=False
resume=False
model_dir=None
outdir=""
eval_conf=""
prec=bf16
base=clip

all_gpus=$(nvidia-smi  -L | wc -l)

function usage
{
    echo "usage: arg_parse_example --dataset --name --ft [--epochs || --bs || --seed || --ema || --help]";
}

function inat2018_dataset
{
    if [ -z "$SLURM_JOB_ID" ]; then
        echo "skip"
    else
        bash scripts/prepare_inat2018.sh
    fi
}

function imagnet_lt_dataset
{
    if [ -z "$SLURM_JOB_ID" ]; then
        echo "skip"
    else
        bash scripts/prepare_imagenet_lt.sh
    fi
}

function domainnet_dataset
{
    if [ -z "$SLURM_JOB_ID" ]; then
        echo "skip"
    else
        bash scripts/prepare_domainnet.sh
    fi
}

function imagnet_shift_dataset
{
    if [ -z "$SLURM_JOB_ID" ]; then
        echo "skip"
    else
        bash scripts/prepare_imagenet_lt.sh
    fi
}

function fewshot_dataset
{
    if [ -z "$SLURM_JOB_ID" ]; then
        echo "skip"
    else
        bash scripts/prepare_imagenet_lt.sh
    fi
}

function imagenet_dataset
{
    if [ -z "$SLURM_JOB_ID" ]; then
        echo "skip"
    else
        bash scripts/prepare_imagenet.sh
    fi
}

function cifar100_dataset
{   
    if [ -z "$SLURM_JOB_ID" ]; then
        echo "skip"
    else
        bash scripts/prepare_cifar100.sh
    fi
}

function parse_args
{

    # positional args
    args=()

    # named args
    while [ "$1" != "" ]; do
        case "$1" in
            --dataset )                   dataset="$2";                 shift 2;;
            --name )                      name="$2";                    shift 2;;
            --ft )                        ft="$2";                      shift 2;;
            --base )                      base="$2";                    shift 2;;
            --epochs )                    epochs="$2";                  shift 2;;
            --bs )                        cluster_bs="$2";              shift 2;;
            --seed )                      seed="$2";                    shift 2;;
            --model_dir )                 model_dir="$2";               shift 2;;
            --outdir)                     outdir="$2";                  shift 2;;
            --eval_conf )                 eval_conf="$2";               shift 2;;
            --ema )                       ema="$2";                     shift 2;;
            --resume )                    resume=True;                  shift;;
            --o3 )                        o3=True;                      shift;;
            --help )                      usage;                        exit;;          # quit and show usage
            * )                           args+=("$1");                 shift;;         # if no match, add it to the positional args
        esac
    done

    # validate required args
    if [[ -z "${name}" || -z "${dataset}" || -z "${ft}" ]]; then
        echo "name, dataset and ft must be set"
        usage
        exit;
    fi

    if [[ "${dataset}" != "tinyimagenet" \
            && "${dataset}" != "tinyimagenet_lt" \
            && "${dataset}" != "tinyimagenetc" \
            && "${dataset}" != "domainnet_c" \
            && "${dataset}" != "domainnet_c_shift" \
            && "${dataset}" != "domainnet_p" \
            && "${dataset}" != "domainnet_p_shift" \
            && "${dataset}" != "domainnet_s" \
            && "${dataset}" != "domainnet_s_shift" \
            && "${dataset}" != "domainnet_r" \
            && "${dataset}" != "domainnet_r_shift" \
            && "${dataset}" != "domainnet_qd" \
            && "${dataset}" != "domainnet_qd_shift" \
            && "${dataset}" != "mini_imagenet" \
            && "${dataset}" != "mini_imagenet_shift" \
            && "${dataset}" != "imagenet" \
            && "${dataset}" != "imagenet_shift" \
            && "${dataset}" != "imagenet_lt" \
            && "${dataset}" != "imagenet_s100" \
            && "${dataset}" != "imagenet_s500" \
            && "${dataset}" != "imagenetc" \
            && "${dataset}" != "cifar100" \
            && "${dataset}" != "cifar100_lt" \
            && "${dataset}" != "cifar100c" \
            && "${dataset}" != "inat2018" \
            && "${dataset}" != "iwildcam_id" \
            && "${dataset}" != "iwildcam_ood" \
            && "${dataset}" != "iwildcam_oracle" \
            && "${dataset}" != "fmow_id" \
            && "${dataset}" != "fmow_ood" \
            && "${dataset}" != "fewshot" ]]; then
        echo "unknown dataset"
        usage
        exit;
    fi

    if [[ "${ft}" != "lp" \
            && "${ft}" != "full" \
            &&  "${ft}" != "lora" \
            &&  "${ft}" != "dora" \
            &&  "${ft}" != "soma" \
            &&  "${ft}" != "adalora" \
            &&  "${ft}" != "pace" \
            && "${ft}" != "mask" \
            && "${ft}" != "maskr" \
            && "${ft}" != "mixout" ]]; then
        echo "ft must be one of: lp/full/lora/dora/soma/adalora/pace/mask/maskr/mixout"
        usage
        exit;
    fi

    if [[ "${base}" != "clip" \
            && "${base}" != "in1k" \
            && "${base}" != "dino" \
            && "${base}" != "in21k" ]]; then
        echo "base must be one of: clip/in1k/dino/in21k"
        usage
        exit;
    fi

    if [[ "${eval_conf}" != "" \
            && "${eval_conf}" != "default" \
            && "${eval_conf}" != "wise" \
            && "${eval_conf}" != "zs" ]]; then
        echo "eval_conf must be one of: /default/wise/zs"
        usage
        exit;
    fi
}

parse_args "$@"

# check if there file DONE exists in the output/${dataset}_${name} directory and resume if True
if [ "$resume" == "True" ]; then
    if [ -f "./output/${dataset}_${name}/DONE" ]; then
        echo "Experiment ${dataset}_${name} is already finished. Exiting."
        exit 0
    fi
fi

start=`date +%s`

if [ -z "$SLURM_JOB_ID" ]; then
    echo "Running on a local machine"

    outdir="${outdir:-./output}"
    data_dir=/export/livia/home/vision/masih/local/masih/dataset

    num_workers=8
    batch_size=$(( 160 < $cluster_bs ? 160 : $cluster_bs ))
    gpu=None
    num_gpus=1
else
    echo "Running on a cluster"

    outdir=$project/workspace/sparse_mask/output
    data_dir=$SLURM_TMPDIR

    num_workers=$(( 8 < $SLURM_CPUS_ON_NODE ? 8 : $SLURM_CPUS_ON_NODE )) # use 16 workers or less if not available
    batch_size=$cluster_bs
    gpu=None
    num_gpus=$all_gpus
fi

if [[ "${dataset}" == "imagenet_lt" ]]; then
    imagnet_lt_dataset
    dataset=imagenet_lt
elif [[ "${dataset}" == "inat2018" ]]; then
    inat2018_dataset
    dataset=inat2018
elif [[ "${dataset}" == "imagenet_shift" ]]; then
    imagnet_shift_dataset
    dataset=imagenet_shift
elif [[ "${dataset}" == "imagenet" ]]; then
    imagenet_dataset
    dataset=imagenet
elif [[ "${dataset}" == "cifar100" ]]; then
    cifar100_dataset
    dataset=cifar100
elif [[ "${dataset}" == "cifar100_lt" ]]; then
    cifar100_dataset
    dataset=cifar100_lt
elif [[ "${dataset}" == "cifar100c" ]]; then
    cifar100_dataset
    dataset=cifar100c
elif [[ "${dataset}" == "domainnet"* ]]; then
    domainnet_dataset
fi

if [ -z "$SLURM_JOB_ID" ]; then
    echo "skip"
else
    # skip rsync if the current directory is already $SLURM_TMPDIR/sparse_mask
    if [ "$PWD" == "$SLURM_TMPDIR/sparse_mask" ]; then
        echo "already in $SLURM_TMPDIR/sparse_mask"
        echo "skip rsync"
    else
        echo "syncing sparse_mask repo in $SLURM_TMPDIR"
        rsync -av ../sparse_mask $SLURM_TMPDIR --exclude output --exclude .venv --exclude .git --exclude slurm_logs
        cd $SLURM_TMPDIR/sparse_mask
    fi

    bash scripts/create_venv_slurm.sh
    echo "activating virtual environment"
    source .venv/bin/activate
    echo "running experiment"
fi

end=`date +%s`
echo "Preprocessing of the job executed in $((end - start)) seconds"

model_config=${base}/${ft}_${base}_vit_b16
echo "Parsed args: name=${name}, dataset=${dataset}, base=${base}, model_config=${model_config}, seed=${seed}, prec=${prec},
        model_dir=${model_dir}, eval_conf=${eval_conf}, data_dir=${data_dir}, outdir=${outdir},
        epochs=${epochs}, ft=${ft}, bs=${batch_size}, ema=${ema}, resume=${resume}, o3=${o3}"

if [ ${#args[@]} -ne 0 ]; then
    echo "Extra args: ${args[@]}"
fi

torchrun --standalone --nnodes=1 --nproc-per-node=$num_gpus \
    main.py -d $dataset -m $model_config -e "$eval_conf" --data_dir $data_dir --outdir $outdir \
    output_dir ${dataset}_${base}_${name} \
    num_workers $num_workers prec $prec \
    gpu $gpu num_epochs $epochs batch_size $batch_size test_batch_size $test_batch_size ema $ema \
    seed $seed \
    tensorboard False resume $resume o3 $o3 model_dir $model_dir "${args[@]}"