#!/bin/bash

cd ../

slurm_pre='--partition t4v2,p100,t4v1,rtx6000 --gres gpu:1 --mem 7gb -c 4 --job-name subsample --output /scratch/ssd001/home/haoran/projects/ClinicalDG/slurm_logs/eicusubsample_%A.log'


for dataset in eICUSubsampleUnobs eICUSubsampleObs; do
    for es_method in train val; do
        architecture='GRU'
        hparams='{"eicu_architecture": '"\"${architecture}\""'}'
        hparams=`echo $hparams | sed 's/\"/\\\"/g'`
        python -m clinicaldg.scripts.sweep launch \
            --output_dir "/scratch/ssd001/home/haoran/clinicaldg_results/eICUSubsample" \
            --command_launcher "slurm" \
            --n_trials 5 \
            --algorithms ERMID ERMMerged ERM IRM VREx RVP IGA CORAL MLDG GroupDRO \
            --datasets ${dataset} \
            --n_hparams 10 \
            --slurm_pre "${slurm_pre}" \
            --es_method "${es_method}" \
            --hparams "${hparams}" \
            --skip_confirmation \
            --delete_model
    done
done
