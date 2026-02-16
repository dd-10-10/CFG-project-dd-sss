#! /usr/bin/env bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate bio

for tf in CTCF REST EP300
do
    for m in {0..10}
    do
	echo "Starting" $m $tf
        python pythonScripts/run_me.py data/tsv/chr4_200bp_bins.tsv data/MidSem_SuchetsRun/ "$m" 5 "$tf" --force_recalculate --delete_temp
    done
done
