#!/bin/bash

echo "preparing ImageNet_LT dataset in $SLURM_TMPDIR"
start=`date +%s`

if [ ! -d $SLURM_TMPDIR/ImageNet/LT_train ]; then
    echo "copying imagenet_LT_train.tar to $SLURM_TMPDIR"
    cp $project/dataset/imagenet_LT_train.tar $SLURM_TMPDIR
    echo "extracting imagenet_LT_train.tar to $SLURM_TMPDIR"
    tar -xf $SLURM_TMPDIR/imagenet_LT_train.tar -C $SLURM_TMPDIR && rm -f $SLURM_TMPDIR/imagenet_LT_train.tar
else
    echo "ImageNet_LT train dataset already exists in $SLURM_TMPDIR, skipping extraction"
fi

if [ ! -d $SLURM_TMPDIR/ImageNet/val ]; then
    echo "copying imagenet_val.tar to $SLURM_TMPDIR"
    cp $project/dataset/imagenet_val.tar $SLURM_TMPDIR
    echo "extracting imagenet_val.tar to $SLURM_TMPDIR"
    tar -xf $SLURM_TMPDIR/imagenet_val.tar -C $SLURM_TMPDIR && rm -f $SLURM_TMPDIR/imagenet_val.tar
else
    echo "ImageNet_LT val dataset already exists in $SLURM_TMPDIR, skipping extraction"
fi

end=`date +%s`
echo "ImageNet_LT dataset prepared in $((end - start)) seconds"