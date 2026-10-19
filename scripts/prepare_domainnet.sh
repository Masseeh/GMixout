#!/bin/bash

echo "preparing DomainNet dataset in $SLURM_TMPDIR"
start=`date +%s`

if [ ! -d $SLURM_TMPDIR/domain_net ]; then
    echo "copying domain_net.tar.gz to $SLURM_TMPDIR"
    cp $project/dataset/domain_net.tar.gz $SLURM_TMPDIR
    echo "extracting domain_net.tar.gz to $SLURM_TMPDIR"
    tar -xzf $SLURM_TMPDIR/domain_net.tar.gz -C $SLURM_TMPDIR && rm -f $SLURM_TMPDIR/domain_net.tar.gz
else
    echo "DomainNet dataset already exists in $SLURM_TMPDIR, skipping extraction"
fi

end=`date +%s`
echo "DomainNet dataset prepared in $((end - start)) seconds"