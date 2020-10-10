"""
Script for preparing COVID=19 data, and performing train-val-test split
"""

# import csv
import os
import numpy as np
import pandas as pd
from collections import Counter 
import random

# Dataset path
COVID_DATA_PATH='./gan_classifier/gan_data_tools/covid-ar-data-reader'
METADATA_CSV = os.path.join(COVID_DATA_PATH, 'covid-ar-data.csv')
TRAIN_FILE = './data/covid19_ar/train_list.txt'
VAL_FILE = './data/covid19_ar/val_list.txt'
TEST_FILE = './data/covid19_ar/test_list.txt'

# Load patient stats
covids = dict()
df = pd.read_csv(METADATA_CSV)

patient_ids = Counter(df['patient_id'].tolist())
covids = {k: v for k, v in sorted(patient_ids.items(), key=lambda item: item[1])}
total_data = sum([v for k,v in covids.items()])
print ("Patient-#X-Rays statistics:")
print (covids)
print ("Total Images:", total_data, '\n')

patient_id_list = list(covids.keys())
random.shuffle(patient_id_list)

total_patient_count = len(covids)
total_test_patient_count = int(total_patient_count * 0.2)
total_val_patient_count = int(total_patient_count * 0.1)

test_patients = set(patient_id_list[0:total_test_patient_count])
val_patients = set(patient_id_list[total_test_patient_count: total_test_patient_count+ total_val_patient_count])


# Assign train-val-test split
# test_patients = set({'4', '15', '86', '59', '6', '82', '80', '78', '76', '65', '36', '32', '50', '18', '115', '152', '138', '70', '116'})
# val_patients = set({'73', '51', '48', '11', '43', '24', '112'})

print ('#Train patients:', len(set(covids.keys()).difference(test_patients.union(val_patients))))
print ('#Test patients:', len(test_patients))
print ('#Val patients:', len(val_patients))
print ()
print ('#Train data points:', sum([v for k, v in covids.items() if k not in test_patients.union(val_patients)]))
print ('#Test data points:', sum([v for k, v in covids.items() if k in test_patients]))
print ('#Val data points:', sum([v for k, v in covids.items() if k in val_patients]))

# Construct the split lists
train_list = []
test_list = []
val_list = []

for i, row in df.iterrows():
    patient_id = row['patient_id']
    filename =  row['filename']

    if patient_id in test_patients:
        test_list.append(filename)
    elif patient_id in val_patients:
        val_list.append(filename)
    else:
        train_list.append(filename)

print (len(train_list), len(test_list), len(val_list))

# Write image list in file
def make_img_list(data_file, img_file_list):
    with open(data_file, 'w') as f:
        for imgfile in img_file_list:
            try: 
                assert os.path.isfile(os.path.join(imgfile))
                #if (os.path.exists(os.path.join(COVID_DATA_PATH, imgfile))):
                f.write("%s\n" % imgfile)
            except: 
                print ("Image %s NOT FOUND" % imgfile)

make_img_list(TRAIN_FILE, train_list)
make_img_list(VAL_FILE, val_list)
make_img_list(TEST_FILE, test_list)
