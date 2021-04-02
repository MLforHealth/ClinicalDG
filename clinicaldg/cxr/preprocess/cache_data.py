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
from clinicaldg.cxr import process
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--env_id', type = int, required = False)
args = parser.parse_args()

validate.validate_splits()

dfs = {}
for env in df_paths:
    func = process.get_process_func(env)
    df_env = func(pd.read_csv(df_paths[env]), only_frontal = True)
    dfs[env] = { 
        'all': df_env
    }

i=0
dss = []
for env in dfs:    
    if args.env_id is None or args.env_id == i:
        ds = data.get_dataset(dfs, envs = [env], split = 'all', augment = -1, cache = True, only_frontal = False)
        print(env)
        dss.append(ds)        
    i += 1

for ds in dss:        
    for i in range(len(ds)):
        ds[i]
