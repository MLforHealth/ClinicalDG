#!/bin/bash

cd ../

slurm_pre='--partition t4v2,p100,t4v1,rtx6000 --gres gpu:1 --mem 40gb -c 4 --job-name cxr --output /scratch/ssd001/home/haoran/projects/ClinicalDG/slurm_logs/cxr_%A.log'

for es_method in train val; do
    hparams='{"cxr_augment": 1, "use_cache": 1}'
    hparams=`echo $hparams | sed 's/\"/\\\"/g'`
    python -m clinicaldg.scripts.sweep launch \
        --output_dir "/scratch/ssd001/home/haoran/clinicaldg_results/CXR" \
        --command_launcher "slurm" \
        --n_trials 5 \
        --algorithms ERMID ERM IRM VREx RVP IGA CORAL MLDG GroupDRO \
        --datasets CXR CXRBinary\
        --n_hparams 10 \
        --slurm_pre "${slurm_pre}" \
        --es_method "${es_method}" \
        --hparams "${hparams}" \
        --skip_confirmation
done
