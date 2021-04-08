# An Empirical Framework for Domain Generalization In Clinical Settings

## Paper
If you use this code in your research, please cite the following publication:
```
@inproceedings{zhang2021empirical,
  title={An empirical framework for domain generalization in clinical settings},
  author={Zhang, Haoran and Dullerud, Natalie and Seyyed-Kalantari, Laleh and Morris, Quaid and Joshi, Shalmali and Ghassemi, Marzyeh},
  booktitle={Proceedings of the Conference on Health, Inference, and Learning},
  pages={279--290},
  year={2021}
}
```

This paper can also be found on arxiv: https://arxiv.org/abs/2103.11163 


## Acknowledgements

Our implementation is a modified version of the excellent [DomainBed](https://github.com/facebookresearch/DomainBed) framework (from commit [a10458a](https://github.com/facebookresearch/DomainBed/tree/a10458a2adfd8aec0fda2d617f710e5044e5dc60)). We also make use of some code from [eICU Benchmarks](https://github.com/mostafaalishahi/eICU_Benchmark).


## To replicate the experiments in the paper:

### Step 0: Environment and Prerequisites
Run the following commands to clone this repo and create the Conda environment:

```
git clone https://github.com/MLforHealth/ClinicalDG.git
cd ClinicalDG/
conda env create -f environment.yml
conda activate clinicaldg
```

### Step 1: Obtaining the Data
See [DataSources.md](DataSources.md) for detailed instructions.

### Step 2: Running Experiments

Experiments can be ran using the same procedure as for the [DomainBed framework](https://github.com/facebookresearch/DomainBed), with a few additional adjustable data hyperparameters which should be passed in as a JSON formatted dictionary.

For example, to train a single model:
```
python -m clinicaldg.scripts.train\
       --algorithm ERM\
       --dataset eICUSubsampleUnobs\
       --es_method val\
       --hparams  '{"eicu_architecture": "GRU", "eicu_subsample_g1_mean": 0.5, "eicu_subsample_g2_mean": 0.05}'\
       --output_dir /path/to/output
```

To sweep a range of datasets, algorithms, and hyperparameters:
```
python -m clinicaldg.scripts.sweep launch\
       --output_dir=/my/sweep/output/path\
       --command_launcher slurm\
       --algorithms ERMID ERM IRM VREx RVP IGA CORAL MLDG GroupDRO \
       --datasets CXR CXRBinary\
       --n_hparams 10\
       --n_trials 5\
       --es_method train\
       --hparams '{"cxr_augment": 1}'
```

A detailed list of `hparams` available for each dataset can be found [here](hparams.md).

We provide the bash scripts used for our main experiments in the `bash_scripts` directory. You will likely need to customize them, along with the launcher, to your compute environment.

### Step 3: Aggregating Results

We provide sample code for creating aggregate results for an experiment in `notebooks/AggResults.ipynb`.



## License
This source code is released under the MIT license, included [here](LICENSE).
