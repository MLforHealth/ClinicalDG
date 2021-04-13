#!/bin/bash

cd ../

slurm_pre='--partition t4v2 --gres gpu:1 --mem 20gb -c 4 --job-name eicu_vary_lambda --output /scratch/ssd001/home/haoran/projects/ClinicalDG/slurm_logs/eicu_lambda_%A.log'

for method in ERM IRM VREx RVP IGA; do
    for lambda in 0 1e-1 1 1e1 1e2 1e3 1e4 1e5; do
        for es_method in train val; do
            architecture='GRU'
            hparams='{"eicu_architecture": '"\"${architecture}\""','"\"${method,,}_lambda\": ${lambda}, \"${method,,}_penalty_anneal_iters\": 0"'}'
            hparams=`echo $hparams | sed 's/\"/\\\"/g'`
            python -m clinicaldg.scripts.sweep launch \
                --output_dir "/scratch/ssd001/home/haoran/clinicaldg_results/eICU_lambda" \
                --command_launcher "slurm" \
                --n_trials 5 \
                --algorithms ${method} \
                --datasets eICU \
                --n_hparams 10 \
                --slurm_pre "${slurm_pre}" \
                --es_method "${es_method}" \
                --hparams "${hparams}" \
                --skip_confirmation \
                --delete_model
        done
    done
done
