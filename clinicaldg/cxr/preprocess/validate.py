from pathlib import Path
import clinicaldg.cxr.Constants as Constants

def validate_mimic():
    img_dir = Path(Constants.image_paths['MIMIC'])
    assert (img_dir/'mimic-cxr-2.0.0-metadata.csv.gz').is_file()
    assert (img_dir/'mimic-cxr-2.0.0-negbio.csv.gz').is_file()
    assert (img_dir/'mimic-cxr-2.0.0-negbio.csv.gz').is_file()
    assert (img_dir/'patients.csv.gz').is_file()    
    assert (img_dir/'files/p19/p19316207/s55102753/31ec769b-463d6f30-a56a7e09-76716ec1-91ad34b6.jpg').is_file()

def validate_cxp():
    img_dir = Path(Constants.image_paths['CXP'])
    if (img_dir/'CheXpert-v1.0').is_dir():
        cxp_subfolder = 'CheXpert-v1.0'
    else:
        cxp_subfolder = 'CheXpert-v1.0-small'
    assert (img_dir/cxp_subfolder/'train.csv').is_file()
    assert (img_dir/cxp_subfolder/'train/patient48822/study1/view1_frontal.jpg').is_file()
    assert (img_dir/cxp_subfolder/'valid/patient64636/study1/view1_frontal.jpg').is_file()

def validate_pad():
    img_dir = Path(Constants.image_paths['PAD'])
    assert (img_dir/'PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv').is_file()
    assert (img_dir/'images-224'/'185566798805711692534207714722577525271_qb3lyn.png').is_file()

def validate_nih():
    img_dir = Path(Constants.image_paths['NIH'])
    assert (img_dir/'Data_Entry_2017.csv').is_file()
    assert (img_dir/'images/00002072_003.png').is_file()

def validate_splits():
    for dataset in Constants.df_paths:
        for split in Constants.df_paths[dataset]:
            assert Path(Constants.df_paths[dataset][split]).is_file()


def validate_all():
    validate_mimic()
    validate_cxp()
    validate_nih()
    validate_pad()
