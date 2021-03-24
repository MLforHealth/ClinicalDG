from clinicaldg.eicu.Constants import *
import numpy as np
import pandas as pd
from clinicaldg import datasets

def corrupt(col, p):
    return np.logical_xor(col, np.random.binomial(n = 1, p = p, size = len(col)))  

def compute_subsample_probability(subset_df, desired_prob, target_name = 'target'):
    pos = subset_df[target_name].sum()
    neg = len(subset_df) - pos
    cur_pos = pos/len(subset_df)

    if cur_pos >= desired_prob:            
        desired_pos = neg * desired_prob/(1 - desired_prob)
        return (1,  1 - desired_pos/pos) # subsample positive samples
    else:
        desired_neg = pos * (1 - desired_prob)/(desired_prob)
        return (0,  1 - desired_neg/neg) # subsample negative samples
    
def aug_f(grp, target, brackets):    
    a,b = brackets[grp]
    if target == a:
        return b
    else:
        return 0                
                
class AddCorrelatedFeature():
    def __init__(self, train_corrupt_dist, train_corrupt_mean, val_corrupt, test_corrupt, feat_name):
        self.train_corrupt_dist = train_corrupt_dist
        self.train_corrupt_mean = train_corrupt_mean
        self.val_corrupt = val_corrupt
        self.test_corrupt = test_corrupt
        self.feat_name = feat_name
        
        self.train_corrupts = [train_corrupt_mean - train_corrupt_dist, train_corrupt_mean,
                      train_corrupt_mean + train_corrupt_dist]        
        assert(all([i >=0 for i in self.train_corrupts]))
    
    def augment(self, reg_mort, reg_pat, envs):
        if len(envs['train']) == 1: # in-distribution exp
            reg_pat[envs['test'][0]][self.feat_name] = corrupt(reg_pat[envs['test'][0]]['target'], self.test_corrupt).astype(float)
        else:
            for i, p in zip(envs['train'], self.train_corrupts):
                reg_pat[i][self.feat_name] = corrupt(reg_pat[i]['target'], p)

            reg_pat[envs['val'][0]][self.feat_name] = corrupt(reg_pat[envs['val'][0]]['target'], self.val_corrupt).astype(float)
            reg_pat[envs['test'][0]][self.feat_name] = corrupt(reg_pat[envs['test'][0]]['target'], self.test_corrupt).astype(float)
        
class Subsample():
    def __init__(self, g1_mean, g2_mean):
        self.g1_mean = g1_mean
        self.g2_mean = g2_mean   
    
    def subsample(self, mort_df, pat_df, g1_mean, g2_mean):     
        pat_df['group_membership'] = pat_df['gender'] == 'Male'
        
        brackets = {}
        brackets[True] = compute_subsample_probability(pat_df[pat_df.group_membership], g1_mean)
        brackets[False] = compute_subsample_probability(pat_df[~pat_df.group_membership], g2_mean)
        
        pat_df['prob'] = pat_df[['group_membership', 'target']].apply(lambda x: aug_f(x['group_membership'], x['target'], brackets), axis = 1)
        pat_df['roll'] = np.random.binomial(1, p =  pat_df['prob'].values, size = len(pat_df))
        
        drop_inds = list(pat_df.index[(pat_df['roll'] == 1)])
        
        pat_df = pat_df[~pat_df.index.isin(drop_inds)]
        
        mort_df = mort_df.loc[~mort_df.index.get_level_values(0).isin(drop_inds)]
        
        return mort_df, pat_df
    
                
    def augment(self, reg_mort, reg_pat, envs): 
        all_envs = [j for i in envs.values() for j in i]
        means = {}
        if len(envs['train']) == 1: # ID test
            means[envs['test'][0]] = (0.1, 0.5)
        else:
            means[envs['train'][0]] = (self.g1_mean, self.g2_mean)
            means[envs['train'][1]] = (self.g1_mean - 0.1, self.g2_mean + 0.05)
            means[envs['train'][2]] = (self.g1_mean - 0.2, self.g2_mean + 0.1)
            means[envs['val'][0]] = (0.3, 0.3)
            means[envs['test'][0]] = (0.1, 0.5)
        
        for env in means:
            reg_mort[env], reg_pat[env] = self.subsample(reg_mort[env], reg_pat[env], means[env][0], means[env][1])
                     
            
class GaussianNoise():        
    def __init__(self, train_corrupt_dist, train_corrupt_mean, val_corrupt, test_corrupt, std, feat_name = 'admissionweight'):
        self.train_corrupt_dist = train_corrupt_dist
        self.train_corrupt_mean = train_corrupt_mean
        self.val_corrupt = val_corrupt
        self.test_corrupt = test_corrupt
        self.feat_name = feat_name
        self.std = std
        
        self.train_corrupts = [train_corrupt_mean - train_corrupt_dist, train_corrupt_mean,
                      train_corrupt_mean + train_corrupt_dist]       

    def add_noise(self, feat_col, pat_df, mean):
        pat_df['signed_target'] = pat_df['target'] *2 - 1
        pat_df['noise'] = np.random.normal(mean, self.std, size = (len(pat_df), )) * pat_df['signed_target']  
                
        feat_col = feat_col.to_frame(0).apply(lambda x: x[0] + pat_df.loc[x.name[0], 'noise'], axis = 1)
        return feat_col
        
    def augment(self, reg_mort, reg_pat, envs):
        feat_name = self.feat_name          
        if len(envs['train']) == 1: # in-distribution exp
            reg_mort[envs['test'][0]][feat_name] = self.add_noise(reg_mort[envs['test'][0]][feat_name], reg_pat[envs['test'][0]], self.test_corrupt)
        else:
            for c, i in enumerate(envs['train']):
                reg_mort[i][feat_name] = self.add_noise(reg_mort[i][feat_name], reg_pat[i], self.train_corrupts[c])

            reg_mort[envs['val'][0]][feat_name] = self.add_noise(reg_mort[envs['val'][0]][feat_name], reg_pat[envs['val'][0]], self.val_corrupt)
            reg_mort[envs['test'][0]][feat_name] = self.add_noise(reg_mort[envs['test'][0]][feat_name], reg_pat[envs['test'][0]], self.test_corrupt)