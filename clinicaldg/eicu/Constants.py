from pathlib import Path

eicu_dir = Path('/scratch/hdd001/projects/ml4h/projects/eicu-crd')
benchmark_dir = Path('/scratch/hdd001/projects/ml4h/projects/eICU_Benchmark')

normal_values = {'Eyes': 4, 'GCS Total': 15, 'Heart Rate': 86, 'Motor': 6, 'Invasive BP Diastolic': 56,
                     'Invasive BP Systolic': 118, 'O2 Saturation': 98, 'Respiratory Rate': 19,
                     'Verbal': 5, 'glucose': 128, 'admissionweight': 81, 'Temperature (C)': 36,
                     'admissionheight': 170, "MAP (mmHg)": 77, "pH": 7.4, "FiO2": 0.21}

ts_cont_features = ['Heart Rate', 'MAP (mmHg)','Invasive BP Diastolic', 'Invasive BP Systolic', 'O2 Saturation', 'Respiratory Rate', 'Temperature (C)', 'glucose', 'FiO2', 'pH']
ts_cat_features = ['GCS Total', 'Eyes', 'Motor', 'Verbal']
static_cont_features = ['admissionheight', 'admissionweight', 'age']
static_cat_features = ['apacheadmissiondx', 'gender']