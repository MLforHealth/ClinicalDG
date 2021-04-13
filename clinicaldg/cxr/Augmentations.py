import numpy as np
import pandas as pd
from clinicaldg.cxr import Constants
from clinicaldg import datasets
from clinicaldg.eicu.Augmentations import compute_subsample_probability, aug_f

def subsample_augment(dfs, g1_mean, g2_mean, g1_dist, g2_dist, target_name = 'Pneumonia'):   
    means = {}
    means[datasets.CXR.TRAIN_ENVS[0]] = (g1_mean + g1_dist/2, g2_mean - g2_dist/2)
    means[datasets.CXR.TRAIN_ENVS[1]] = (g1_mean - g1_dist/2, g2_mean + g2_dist/2)
    means[datasets.CXR.VAL_ENV] = (0.07, 0.04)
    means[datasets.CXR.TEST_ENV] = (0.05, 0.05)

    print('Subsampling parameters: \n' + str(means), flush = True)

    for env in means:
        for i in means[env]:
            assert(0 <= i <= 1)
    
    new_dfs = {}
    for env in means:
        new_dfs[env] = {}
        for split in dfs[env]:
            df = dfs[env][split]
            
            df['group_membership'] = df['Sex'] == 'M'
        
            brackets = {}
            brackets[True] = compute_subsample_probability(df[df.group_membership], means[env][0], target_name = target_name)
            brackets[False] = compute_subsample_probability(df[~df.group_membership], means[env][1], target_name = target_name)
            
            df['prob'] = df[['group_membership', target_name]].apply(lambda x: aug_f(x['group_membership'], x[target_name], brackets), axis = 1)
            df['roll'] = np.random.binomial(1, p = df['prob'].values, size = len(df))

            df = df[df['roll'] == 0]
            df = df.drop(columns = ['prob', 'roll', 'group_membership'])
            
            new_dfs[env][split] = df
            
    return new_dfs