import numpy as np
import pandas as pd
from clinicaldg.cxr import Constants
from clinicaldg.cxr import process
from clinicaldg import datasets
from clinicaldg.eicu.Augmentations import compute_subsample_probability, aug_f

def subsample_augment(g1_mean, g2_mean, g1_dist, g2_dist, target_name = 'Pneumonia'):   
    means = {}
    means['MIMIC'] = (g1_mean + g1_dist/2, g2_mean - g2_dist/2)
    means['CXP'] = (g1_mean - g1_dist/2, g2_mean + g2_dist/2)
    means['NIH'] = (0.07, 0.04)
    means['PAD'] = (0.05, 0.05)

    for env in means:
        for i in means[env]:
            assert(0 <= i <= 1)
        
    dfs = {}
    for env in Constants.df_paths:
        for split in Constants.df_paths[env]:
            func = process.get_process_func(env)
            if env not in dfs:
                dfs[env] = {}
            df = func(pd.read_csv(Constants.df_paths[env][split]), only_frontal = True)
            
            df['group_membership'] = df['Sex'] == 'M'
        
            brackets = {}
            brackets[True] = compute_subsample_probability(df[df.group_membership], means[env][0], target_name = target_name)
            brackets[False] = compute_subsample_probability(df[~df.group_membership], means[env][1], target_name = target_name)
            
            df['prob'] = df[['group_membership', target_name]].apply(lambda x: aug_f(x['group_membership'], x[target_name], brackets), axis = 1)
            df['roll'] = np.random.binomial(1, p =  df['prob'].values, size = len(df))

            df = df[df['roll'] == 0]
            df = df.drop(columns = ['prob', 'roll', 'group_membership'])
            
            dfs[env][split] = df
            
    return dfs