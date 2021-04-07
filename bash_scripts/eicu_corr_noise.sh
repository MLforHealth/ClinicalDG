#!/bin/bash

cd ../

slurm_pre='--partition t4v2,p100,t4v1,rtx6000 --gres gpu:1 --mem 7gb -c 4 --job-name eicu_gaussian --output /scratch/ssd001/home/haoran/projects/ClinicalDG/slurm_logs/eicu_gaussian_%A.log'

for train_noise in 1.0 2.0; do
    for train_dist in 0.5 1.0; do
        for es_method in train val; do
            architecture='GRU'
            hparams='{"eicu_architecture": '"\"${architecture}\""', "corr_noise_train_corrupt_mean":'"${train_noise}"', "corr_noise_train_corrupt_dist":'"${train_dist}"'}'
            hparams=`echo $hparams | sed 's/\"/\\\"/g'`
            python -m clinicaldg.scripts.sweep launch \
                --output_dir "/scratch/ssd001/home/haoran/clinicaldg_results/eICUGaussianNoise" \
                --command_launcher "slurm" \
                --n_trials 5 \
                --algorithms ERMID ERMMerged ERM IRM VREx RVP IGA CORAL MLDG GroupDRO \
                --datasets eICUCorrNoise \
                --n_hparams 10 \
                --slurm_pre "${slurm_pre}" \
                --es_method "${es_method}" \
                --hparams "${hparams}" \
                --skip_confirmation \
                --delete_model
        done
    done
done
