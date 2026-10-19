#!/bin/bash

echo "preparing ImageNet dataset in $SLURM_TMPDIR"
start=`date +%s`

if [ ! -d $SLURM_TMPDIR/ImageNet/val ]; then
    echo "copying imagenet_val.tar to $SLURM_TMPDIR"
    cp $project/dataset/imagenet_val.tar $SLURM_TMPDIR
    echo "extracting imagenet_val.tar to $SLURM_TMPDIR"
    tar -xf $SLURM_TMPDIR/imagenet_val.tar -C $SLURM_TMPDIR && rm -f $SLURM_TMPDIR/imagenet_val.tar
else
    echo "ImageNet val dataset already exists in $SLURM_TMPDIR, skipping extraction"
fi

if [ ! -d $SLURM_TMPDIR/ImageNet/train ]; then
    echo "copying imagenet_train.tar to $SLURM_TMPDIR"

    # check if nibi is in $SLURM_SUBMIT_HOST
    if [[ $SLURM_SUBMIT_HOST == *"nibi"* ]]; then
        echo "Running on nibi"
        # Create train directory; copy .tar file; change directory
        mkdir $SLURM_TMPDIR/ImageNet/train && cp $project/dataset/imagenet_train.tar $SLURM_TMPDIR/ImageNet/train/ && cd $SLURM_TMPDIR/ImageNet/train/
        
        echo "extracting imagenet_train.tar to $SLURM_TMPDIR"
        # Extract training set; remove compressed file
        tar -xf imagenet_train.tar && rm -f imagenet_train.tar

        echo "extracting imagenet_train subfolders to $SLURM_TMPDIR"
        # For each .tar file: 
        #   1. create directory with same name as .tar file
        #   2. extract and copy contents of .tar file into directory
        #   3. remove .tar file
        find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
    else
        echo "Running on $SLURM_SUBMIT_HOST"
        cp $project/dataset/imagenet_train.tar $SLURM_TMPDIR
        echo "extracting imagenet_train.tar to $SLURM_TMPDIR"
        tar -xf $SLURM_TMPDIR/imagenet_train.tar -C $SLURM_TMPDIR && rm -f $SLURM_TMPDIR/imagenet_train.tar
    fi
else
    echo "ImageNet train dataset already exists in $SLURM_TMPDIR, skipping extraction"
fi

end=`date +%s`
echo "ImageNet dataset prepared in $((end - start)) seconds"