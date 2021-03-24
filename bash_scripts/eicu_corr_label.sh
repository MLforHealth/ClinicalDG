#!/bin/bash

cd ../

slurm_pre='--partition t4v2 --gres gpu:1 --mem 20gb -c 4 --job-name corr_label --output /scratch/ssd001/home/haoran/projects/ClinicalDG/slurm_logs/eicu_corr_%A.log'


for train_corrupt in $(seq 0.10 0.20 0.50); do
    for es_method in train val; do
        architecture='GRU'
        hparams='{"eicu_architecture": '"\"${architecture}\""', "eicu_add_corr_train_corrupt_dist": 0.1, "eicu_add_corr_val_corrupt": 0.5, "eicu_add_corr_test_corrupt": 0.9, "eicu_add_corr_train_corrupt_mean":'"${train_corrupt}"'}'
        hparams=`echo $hparams | sed 's/\"/\\\"/g'`
        python -m clinicaldg.scripts.sweep launch \
            --output_dir "/scratch/ssd001/home/haoran/clinicaldg_results/eICUAddCorr" \
            --command_launcher "slurm" \
            --n_trials 5 \
            --algorithms ERMID ERM IRM VREx RVP CORAL MLDG GroupDRO \
            --datasets eICUCorrLabel \
            --n_hparams 10 \
            --slurm_pre "${slurm_pre}" \
            --es_method "${es_method}" \
            --hparams "${hparams}" \
            --skip_confirmation
    done
done
