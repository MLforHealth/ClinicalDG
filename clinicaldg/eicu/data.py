import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
from clinicaldg.eicu.data_extraction.data_extraction_mortality import data_extraction_mortality
import clinicaldg.eicu.Constants as Constants
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import ConcatDataset, Dataset

hospitals = pd.read_csv((Constants.eicu_dir/'hospital.csv'))
hospitals['region'] = hospitals['region'].fillna('Missing')
patients = pd.read_csv((Constants.eicu_dir/'patient.csv'))[['patientunitstayid', 'hospitalid',  'gender']]

class LabelEncoderExt(object):
    '''
    Label encoder, but when encountering an unseen label on the test set, will set to "Missing"
    '''    
    def __init__(self):
        self.label_encoder = LabelEncoder()

    def fit(self, data_list):
        self.label_encoder = self.label_encoder.fit(list(map(str, list(data_list))) + ['Missing'])
        self.classes_ = self.label_encoder.classes_
        
        return self

    def transform(self, data_list):
        data_list = list(map(str, list(data_list)))
        for unique_item in np.unique(data_list):
            if unique_item not in self.label_encoder.classes_:
                data_list = ['Missing' if x==unique_item else x for x in data_list]

        return self.label_encoder.transform(data_list)

class AugmentedDataset():
    def __init__(self, train_envs, val_env, test_env, augs = [], train_pct = 0.7, val_pct = 0.1, split_test_env = False):
        self.envs = {
            'train': train_envs,
            'val': [val_env],
            'test': [test_env]
        }
        
        self.reg_mort, self.reg_pat, self.scalers, self.labelencoders = self._get_mortality_data(train_pct, val_pct, split_test_env)   
        for a in augs:
            a.augment(self.reg_mort, self.reg_pat, self.envs)        
    
    def get_torch_dataset(self, env, dset):
        '''
        env: a list of region names, or one of ['train', 'val', 'test']
        dset: one of ['train', 'val', 'test']. For the test environment, use "test" for dset
        '''
        if env in ['train', 'val', 'test']:
            regions = self.envs[env]
        else:
            regions = env
        
        datasets = []
        for r in regions:
            datasets.append(eICUDataset(self.reg_mort[r][self.reg_mort[r]['fold'] == dset], self.reg_pat[r][self.reg_pat[r]['fold'] == dset]))
        
        return ConcatDataset(datasets)        
    
    def get_num_levels(self):        
        return ({i: len(self.labelencoders[i].classes_) for i in Constants.ts_cat_features}, 
                {i: len(self.labelencoders[i].classes_) for i in Constants.static_cat_features}, 
               )    
    
    def _get_mortality_data(self, train_pct, val_pct, split_test_env):
        mort_df = data_extraction_mortality(str(Constants.benchmark_dir))

        targets = mort_df.groupby('patientunitstayid').agg({'hospitaldischargestatus': 'first'}).reset_index()
        pat_df = pd.merge(patients, hospitals, on = 'hospitalid', how = 'left')
        pat_df = pd.merge(pat_df, targets, on = 'patientunitstayid', how = 'inner').rename(columns = {'hospitaldischargestatus': 'target'})
                
        pat_df = pat_df[pat_df.patientunitstayid.isin(mort_df.patientunitstayid)].sample(frac = 1)  # shuffle
        # train test split in each env        
        pat_df['fold'] = ''
        pat_df['fold'].iloc[:int(len(pat_df)*train_pct)] = 'train'
        pat_df['fold'].iloc[int(len(pat_df)*train_pct):int(len(pat_df)*(train_pct + val_pct))] = 'val'
        pat_df['fold'].iloc[int(len(pat_df)*(train_pct + val_pct)):] = 'test'
                 
        if not split_test_env:
            for reg in self.envs['val']:
                # validation environment not used for training
                pat_df.loc[(pat_df.region == reg) & (pat_df.fold == 'train'), 'fold'] = 'val' 

            for reg in self.envs['test']:
                pat_df.loc[pat_df.region == reg, 'fold'] = 'test' # everything in test environment is used for test

        mort_df = mort_df.merge(pat_df[['patientunitstayid', 'fold']], on = 'patientunitstayid')
        
        # make sure everyone has exactly 48h hours of data
        ## make multiindex with 48h
        ## groupby and ffill
        ## fill any remaining missing features with normal_values
        iterables = [np.unique(mort_df['patientunitstayid']), list(range(1, mort_df.itemoffset.max()+1))]
        multiind = pd.MultiIndex.from_product(iterables, names = ['patientunitstayid', 'itemoffset'])
        ind_df = pd.DataFrame(index = multiind)
        mort_df = pd.merge(ind_df, mort_df, left_index = True, right_on = ['patientunitstayid', 'itemoffset'], how = 'left')
        
        mort_df = mort_df.set_index(['patientunitstayid', 'itemoffset']).sort_index().groupby('patientunitstayid').ffill()
        
        for back_col in ['hospitaldischargestatus', 'fold'] + Constants.static_cont_features + Constants.static_cat_features:
            mort_df[back_col] = mort_df[back_col].fillna(method = 'backfill')   

        for feat, val in Constants.normal_values.items():
            mort_df[feat] = mort_df[feat].fillna(val)   
            
            
        # scale continuous and static ts features
        scalers = {}    
        for feat in Constants.ts_cont_features + Constants.static_cont_features:
            scalers[feat] = StandardScaler().fit(mort_df.loc[mort_df.fold == 'train', feat].values.reshape(-1, 1))
            mort_df[feat] = scalers[feat].transform(mort_df[feat].values.reshape(-1, 1))[:, 0]
            
        # encode continuous and static cat features  
        labelencoders, num_encodings = {}, {}
        for feat in Constants.ts_cat_features + Constants.static_cat_features:
            mort_df[feat] = mort_df[feat].fillna('Missing')
            labelencoders[feat] = LabelEncoderExt().fit(mort_df.loc[mort_df.fold == 'train', feat])
            mort_df[feat] = labelencoders[feat].transform(mort_df[feat])
            num_encodings[feat] = len(labelencoders[feat].classes_)
            
        reg_mort, reg_pat = {}, {}
        for reg in pat_df.region.unique():
            sub_pat = pat_df[pat_df.region == reg]
            sub = mort_df[mort_df.index.get_level_values(0).isin(sub_pat.patientunitstayid)]

            reg_mort[reg] = sub
            reg_pat[reg] = sub_pat.set_index('patientunitstayid')

        return reg_mort, reg_pat, scalers, labelencoders
           
class eICUDataset(Dataset):
    def __init__(self, mort_df, pat_df):
        self.mort_df = mort_df
        self.pat_df = pat_df
    
    def __len__(self):
        return self.pat_df.shape[0]
    
    def __getitem__(self, idx):
        pat_id = self.pat_df.index[idx]
        mort_data = self.mort_df.loc[pat_id]
        ts_cont_feats = mort_data[Constants.ts_cont_features].values
        ts_cat_feats = mort_data[Constants.ts_cat_features].values
        
        static_not_in_mort = [i for i in Constants.static_cont_features if i not in self.mort_df]
        static_in_mort = [i for i in Constants.static_cont_features if i in self.mort_df]
                
        static_cont_feats = np.concatenate((mort_data[static_in_mort].iloc[0].values, self.pat_df.loc[pat_id, static_not_in_mort].values)).astype(float)
        static_cat_feats = mort_data[Constants.static_cat_features].iloc[0].values     
        
        return ({'pat_id': pat_id,
                'ts_cont_feats': ts_cont_feats,
                'ts_cat_feats': ts_cat_feats,
                'static_cont_feats': static_cont_feats,
                'static_cat_feats': static_cat_feats,
                'gender': int(self.pat_df.loc[pat_id, 'gender'].strip() == 'Male')}, 
            self.pat_df.loc[pat_id, 'target'])