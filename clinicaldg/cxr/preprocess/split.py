import clinicaldg.cxr.Constants as Constants
import numpy as np
import pandas as pd
from pathlib import Path
import os
import sys
import matplotlib.pyplot as plt
from PIL import Image
import json
import random
from clinicaldg.cxr.preprocess.validate import validate_all

def split(subject_df, split_portions = (0.8, 0.9), seed = 42):
    np.random.seed(seed)
    random.seed(seed)

    subject_df['random_number'] = np.random.uniform(size=len(subject_df))

    train_id = subject_df[subject_df['random_number'] <= split_portions[0]].drop(columns=['random_number'])
    valid_id = subject_df[(subject_df['random_number'] > split_portions[0]) & (subject_df['random_number'] <= split_portions[1])].drop(columns=['random_number'])
    test_id = subject_df[subject_df['random_number'] > split_portions[1]].drop(columns=['random_number'])

    return train_id, valid_id, test_id

def split_mimic():
    img_dir = Path(Constants.image_paths['MIMIC'])
    out_folder = img_dir/'clinicaldg_split'
    out_folder.mkdir(parents = True, exist_ok = True)  

    patients = pd.read_csv(img_dir/'patients.csv.gz')
    labels = pd.read_csv(img_dir/'mimic-cxr-2.0.0-negbio.csv.gz')
    meta = pd.read_csv(img_dir/'mimic-cxr-2.0.0-metadata.csv.gz')

    df = meta.merge(patients, on = 'subject_id').merge(labels, on = ['subject_id', 'study_id'])
    df['age_decile'] = pd.cut(df['anchor_age'], bins = list(range(0, 100, 10))).apply(lambda x: f'{x.left}-{x.right}').astype(str)
    df['frontal'] = df.ViewPosition.isin(['AP', 'PA'])

    df['path'] = df.apply(lambda x: os.path.join('files', f'p{str(x["subject_id"])[:2]}', f'p{x["subject_id"]}', f's{x["study_id"]}', f'{x["dicom_id"]}.jpg'), axis = 1)

    subject_df = pd.DataFrame({'subject_id': np.sort(df['subject_id'].unique())})

    train_id, valid_id, test_id = split(subject_df)

    train_df = df[df.subject_id.isin(train_id.subject_id)]
    valid_df = df[df.subject_id.isin(valid_id.subject_id)]
    test_df = df[df.subject_id.isin(test_id.subject_id)]   

    train_df.to_csv(out_folder/"train.csv", index=False)
    valid_df.to_csv(out_folder/"val.csv", index=False)
    test_df.to_csv(out_folder/"test.csv", index=False)


def split_pad():
    pad_dir = Path(Constants.image_paths['PAD'])
    out_folder = pad_dir/'clinicaldg_split'
    out_folder.mkdir(parents = True, exist_ok = True)         

    df = pd.read_csv(pad_dir/'PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv')
    df = df[['ImageID', 'StudyID', 'PatientID','PatientBirth','PatientSex_DICOM', 'ViewPosition_DICOM', 'Projection','Labels']]
    df = df[~df["Labels"].isnull()]
    df = df[df["ImageID"].apply(lambda x: os.path.exists(os.path.join(pad_dir, 'images-224', x)))]
    df = df[df.Projection.isin(['PA', 'L', 'AP_horizontal', 'AP'])]

    df['frontal'] = ~(df['Projection'] == 'L')
    df = df[~df['Labels'].apply(lambda x: 'exclude' in x or 'unchanged' in x)]

    mapping = dict()

    mapping['Effusion'] = ['hydropneumothorax', 'empyema', 'hemothorax']
    mapping["Consolidation"] = ["air bronchogram"]
    mapping['No Finding'] = ['normal']

    for pathology in Constants.take_labels:
        mask = df["Labels"].str.contains(pathology.lower())
        if pathology in mapping:
            for syn in mapping[pathology]:
                mask |= df["Labels"].str.contains(syn.lower())
        df[pathology] = mask.astype(int)

    df['Age'] = 2017 - df['PatientBirth']

    unique_patients = df['PatientID'].unique()

    np.random.seed(42)
    random.seed(42)

    np.random.shuffle(unique_patients)

    train, val, test = np.split(unique_patients, [int(.8*len(unique_patients)), int(.9*len(unique_patients))])
    train_df = df[df.PatientID.isin(train)]
    val_df = df[df.PatientID.isin(val)]
    test_df = df[df.PatientID.isin(test)]

    train_df.reset_index(drop = True).to_csv(out_folder/'train.csv', index = False)
    val_df.reset_index(drop = True).to_csv(out_folder/'val.csv', index = False)
    test_df.reset_index(drop = True).to_csv(out_folder/'test.csv', index = False)


def split_cxp():
    img_dir = Path(Constants.image_paths['CXP'])
    out_folder = img_dir/'clinicaldg_split'
    out_folder.mkdir(parents = True, exist_ok = True)  
    df = pd.read_csv(img_dir/"map.csv")

    df['subject_id'] = df['Path'].apply(lambda x: int(Path(x).parent.parent.name[7:]))

    subject_df = pd.DataFrame({'subject_id': np.sort(df['subject_id'].unique())})

    train_id, valid_id, test_id = split(subject_df)

    train_df = df[df.subject_id.isin(train_id.subject_id)]
    valid_df = df[df.subject_id.isin(valid_id.subject_id)]
    test_df = df[df.subject_id.isin(test_id.subject_id)]   

    train_df.to_csv(out_folder/"train.csv", index=False)
    valid_df.to_csv(out_folder/"val.csv", index=False)
    test_df.to_csv(out_folder/"test.csv", index=False)


def split_nih():
    img_dir = Path(Constants.image_paths['NIH'])
    out_folder = img_dir/'clinicaldg_split'
    out_folder.mkdir(parents = True, exist_ok = True)  
    df = pd.read_csv(img_dir/"Data_Entry_2017.csv")
    df['labels'] = df['Finding Labels'].apply(lambda x: x.split('|'))

    for label in Constants.take_labels:
        df[label] = df['labels'].apply(lambda x: label in x)

    subject_df = pd.DataFrame({'Patient ID': np.sort(df['Patient ID'].unique())})

    train_id, valid_id, test_id = split(subject_df)
    train_df = df[df['Patient ID'].isin(train_id['Patient ID'])]
    valid_df = df[df['Patient ID'].isin(valid_id['Patient ID'])]
    test_df = df[df['Patient ID'].isin(test_id['Patient ID'])]   

    train_df.to_csv(out_folder/"train.csv", index=False)
    valid_df.to_csv(out_folder/"val.csv", index=False)
    test_df.to_csv(out_folder/"test.csv", index=False)






if __name__ == '__main__':
    print("Validating paths...")
    validate_all()
    print("Splitting MIMIC-CXR...")
    split_mimic()
    print("Splitting CheXpert...")
    split_cxp()
    print("Splitting ChestX-ray8...")
    split_nih()
    print("Splitting PadChest... This might take a few minutes...")
    split_pad()