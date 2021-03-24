import os

#-------------------------------------------
image_paths = {
    'MIMIC': '/scratch/hdd001/projects/ml4h/projects/mimic_access_required/MIMIC-CXR-JPG', # MIMIC-CXR
    'CXP': '/scratch/hdd001/projects/ml4h/projects/CheXpert/', # CheXpert
    'NIH': '/scratch/hdd001/projects/ml4h/projects/NIH/', # ChestX-ray8
    'PAD': '/scratch/hdd001/projects/ml4h/projects/padchest/PADCHEST/', # PadChest
}

cache_dir = '/scratch/ssd001/home/haoran/projects/IRM_Clinical/cache'

#-------------------------------------------

df_paths = {
    dataset: {
        split: os.path.join(image_paths[dataset], 'clinicaldg_split', f'{split}.csv')
        for split in ['train', 'val', 'test']
    } for dataset in image_paths 
}

take_labels = ['No Finding', 'Atelectasis', 'Cardiomegaly',  'Effusion',  'Pneumonia', 'Pneumothorax', 'Consolidation','Edema' ]

IMAGENET_MEAN = [0.485, 0.456, 0.406]         # Mean of ImageNet dataset (used for normalization)
IMAGENET_STD = [0.229, 0.224, 0.225]          # Std of ImageNet dataset (used for normalization)
