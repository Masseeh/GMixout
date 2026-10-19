#!/bin/bash

echo "preparing CIFAR100 dataset in $SLURM_TMPDIR"
start=`date +%s`

if [ ! -d $SLURM_TMPDIR/CIFAR100 ]; then
    echo "copying CIFAR100.tar to $SLURM_TMPDIR"
    cp $project/dataset/CIFAR100.tar $SLURM_TMPDIR
    echo "extracting CIFAR100.tar to $SLURM_TMPDIR"
    tar -xf $SLURM_TMPDIR/CIFAR100.tar -C $SLURM_TMPDIR && rm -f $SLURM_TMPDIR/CIFAR100.tar
else
    echo "CIFAR100 dataset already exists in $SLURM_TMPDIR, skipping extraction"
fi

end=`date +%s`
echo "CIFAR100 dataset prepared in $((end - start)) seconds"