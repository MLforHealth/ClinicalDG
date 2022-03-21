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

def preprocess_mimic():
    img_dir = Path(Constants.image_paths['MIMIC'])
    out_folder = img_dir/'clinicaldg'
    out_folder.mkdir(parents = True, exist_ok = True)  

    patients = pd.read_csv(img_dir/'patients.csv.gz')
    labels = pd.read_csv(img_dir/'mimic-cxr-2.0.0-negbio.csv.gz')
    meta = pd.read_csv(img_dir/'mimic-cxr-2.0.0-metadata.csv.gz')

    df = meta.merge(patients, on = 'subject_id').merge(labels, on = ['subject_id', 'study_id'])
    df['age_decile'] = pd.cut(df['anchor_age'], bins = list(range(0, 100, 10))).apply(lambda x: f'{x.left}-{x.right}').astype(str)
    df['frontal'] = df.ViewPosition.isin(['AP', 'PA'])

    df['path'] = df.apply(lambda x: os.path.join('files', f'p{str(x["subject_id"])[:2]}', f'p{x["subject_id"]}', f's{x["study_id"]}', f'{x["dicom_id"]}.jpg'), axis = 1)
    df.to_csv(out_folder/"preprocessed.csv", index=False)

def preprocess_pad():
    pad_dir = Path(Constants.image_paths['PAD'])
    out_folder = pad_dir/'clinicaldg'
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
    df.reset_index(drop = True).to_csv(out_folder/"preprocessed.csv", index=False)


def preprocess_cxp():
    img_dir = Path(Constants.image_paths['CXP'])
    out_folder = img_dir/'clinicaldg'
    if (img_dir/'CheXpert-v1.0'/'train.csv').is_file():
        df = pd.concat([pd.read_csv(img_dir/'CheXpert-v1.0'/'train.csv'), 
                        pd.read_csv(img_dir/'CheXpert-v1.0'/'valid.csv')],
                        ignore_index = True)
    elif (img_dir/'CheXpert-v1.0-small'/'train.csv').is_file(): 
        df = pd.concat([pd.read_csv(img_dir/'CheXpert-v1.0-small'/'train.csv'),
                        pd.read_csv(img_dir/'CheXpert-v1.0-small'/'valid.csv')],
                        ignore_index = True)
    elif (img_dir/'train.csv').is_file():
        raise ValueError('Please set Constants.image_paths["CXP"] to be the PARENT of the current'+
                ' directory and rerun this script.')
    else:
        raise ValueError("CheXpert files not found!")

    out_folder.mkdir(parents = True, exist_ok = True)  

    df['subject_id'] = df['Path'].apply(lambda x: int(Path(x).parent.parent.name[7:]))
    df.reset_index(drop = True).to_csv(out_folder/"preprocessed.csv", index=False)

def preprocess_nih():
    img_dir = Path(Constants.image_paths['NIH'])
    out_folder = img_dir/'clinicaldg'
    out_folder.mkdir(parents = True, exist_ok = True)  
    df = pd.read_csv(img_dir/"Data_Entry_2017.csv")
    df['labels'] = df['Finding Labels'].apply(lambda x: x.split('|'))

    for label in Constants.take_labels:
        df[label] = df['labels'].apply(lambda x: label in x)
    df.reset_index(drop = True).to_csv(out_folder/"preprocessed.csv", index=False)


if __name__ == '__main__':
    print("Validating paths...")
    validate_all()
    print("Preprocessing MIMIC-CXR...")
    preprocess_mimic()
    print("Preprocessing CheXpert...")
    preprocess_cxp()
    print("Preprocessing ChestX-ray8...")
    preprocess_nih()
    print("Preprocessing PadChest... This might take a few minutes...")
    preprocess_pad()
