#!/h/haoran/anaconda3/bin/python
#SBATCH --partition cpu
#SBATCH --qos nopreemption
#SBATCH -c 4
#SBATCH --mem=20gb

import os
import sys
sys.path.append(os.getcwd())
from pathlib import Path
from clinicaldg.cxr.Constants import *
import numpy as np
from clinicaldg.cxr import data
from clinicaldg.cxr.preprocess import validate
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--split', type = int, required = False)
args = parser.parse_args()

validate.validate_splits()

i=0
dss = []
for env in list(image_paths.keys()):
    for split in ['train', 'val', 'test']:        
        if args.split is None or args.split == i:
            ds = data.get_dataset(envs = [env], split = split, augment = -1, cache = True, only_frontal = False)
            print(env, split)
            dss.append(ds)        
        i += 1

for ds in dss:        
    for i in range(len(ds)):
        ds[i]
