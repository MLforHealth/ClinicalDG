# eICU Data

1. [Obtain access](https://eicu-crd.mit.edu/gettingstarted/access/) to the eICU Collaborative Research Database on PhysioNet and download the [dataset](https://physionet.org/content/eicu-crd/2.0/).


2. Clone the [eICU Benchmarks](https://github.com/mostafaalishahi/eICU_Benchmark) repository and follow the instructions under the "Data extraction" section.

3. Update the `eicu_dir` and `benchmark_dir` variables in `clinicaldg/eicu/Constants.py` to point to the raw data and processed data folders.

# Chest X-ray Data
## Downloading the Data
### MIMIC-CXR
1. [Obtain access](https://mimic-cxr.mit.edu/about/access/) to the MIMIC-CXR-JPG Database Database on PhysioNet and download the [dataset](https://physionet.org/content/mimic-cxr-jpg/2.0.0/). We recommend downloading from the GCP bucket:

```
gcloud auth login
mkdir MIMIC-CXR-JPG
gsutil -m rsync -d -r gs://mimic-cxr-jpg-2.0.0.physionet.org MIMIC-CXR-JPG
```

2. In order to obtain gender information for each patient, you will need to obtain access to [MIMIC-IV](https://physionet.org/content/mimiciv/0.4/). Download `core/patients.csv.gz` and place the file in the `MIMIC-CXR-JPG` directory.

### CheXpert
1. Sign up with your email address [here](https://stanfordmlgroup.github.io/competitions/chexpert/).

2. Download either the original or the downsampled dataset (we recommend the downsampled version - `CheXpert-v1.0-small.zip`) and extract it.


### ChestX-ray8

1. Download the `images` folder and `Data_Entry_2017_v2020.csv` from the [NIH website](https://nihcc.app.box.com/v/ChestXray-NIHCC).

2. Unzip all of the files in the `images` folder.



### PadChest

1. We use a resized version of PadChest, which can be downloaded [here](https://academictorrents.com/details/96ebb4f92b85929eadfb16761f310a6d04105797).

2. Unzip `images-224.tar`.


## Data Processing
1. In `clinicaldg/cxr/Constants.py`, update `image_paths` to point to each of the four directories that you downloaded.

2. Run `python -m clinicaldg.cxr.preprocess.preprocess`. 

3. (Optional) If you are training a lot of models, it _might_ be faster to cache all images to binary 224x224 files on disk. In this case, you should update the `cache_dir` path in `clinicaldg/cxr/Constants.py` and then run `python -m clinicaldg.cxr.preprocess.cache_data`, optionally parallelizing over `--env_id {0, 1, 2, 3}` for speed. To use the cached files, pass `--use_cache 1` to `train.py` or `sweep.py`.


# Colored MNIST

1. Update `mnist_dir` in `clinicaldg/scripts/download.py`.

2. Run `python -m clinicaldg.scripts.download`.