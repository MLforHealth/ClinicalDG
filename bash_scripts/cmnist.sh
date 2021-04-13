#!/bin/bash

cd ../

slurm_pre='--partition t4v2 --gres gpu:1 --mem 20gb -c 4 --job-name cmnist --output /scratch/ssd001/home/haoran/projects/ClinicalDG/slurm_logs/cmnist_%A.log'

lst_hparams=()

for eta in $(seq 0.0 0.05 0.5); do
    hparams='{"cmnist_eta": '"${eta}"', "cmnist_delta": 0.1, "cmnist_beta": 0.15}'
    hparams=`echo $hparams | sed 's/\"/\\\"/g'`
    lst_hparams+=("$hparams")
done

for beta in $(seq 0.05 0.05 0.5); do
    hparams='{"cmnist_eta": 0.25, "cmnist_delta": 0.1, "cmnist_beta":'"${beta}"'}'
    hparams=`echo $hparams | sed 's/\"/\\\"/g'`
    lst_hparams+=("$hparams")
done

for delta in $(seq 0.0 0.05 0.3); do
    hparams='{"cmnist_eta": 0.25, "cmnist_delta": '"${delta}"', "cmnist_beta": 0.15}'
    hparams=`echo $hparams | sed 's/\"/\\\"/g'`
    lst_hparams+=("$hparams")
done

for hparams in "${lst_hparams[@]}"; do
   for es_method in val train; do
        echo ${hparams}
        python -m clinicaldg.scripts.sweep launch \
            --output_dir "/scratch/ssd001/home/haoran/clinicaldg_results/CMNIST" \
            --command_launcher "slurm" \
            --n_trials 5 \
            --algorithms ERM IRM VREx IGA RVP GroupDRO CORAL MLDG  \
            --datasets ColoredMNIST \
            --n_hparams 20 \
            --slurm_pre "${slurm_pre}" \
            --es_method "${es_method}" \
            --hparams "${hparams}" \
            --skip_confirmation \
            --delete_model
    done
done