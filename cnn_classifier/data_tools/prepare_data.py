"""
Script to prepare combined dataset
Class 0: Normal
Class 1: Bacterial Pneumonia
Class 2: Viral Pneumonia
Class 3: COVID-19
"""
import glob
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--combine_pneumonia", action='store_true', default=False)
parser.add_argument("--use_generated", type=bool, default=False)
args = parser.parse_args()

COVID19_DATA_PATH = "./data/covid19"
COVID19_AR_DATA_PATH = "./data/covid19_ar"
PNEUMONIA_DATA_PATH = "./data/Data/chest-xray-pneumonia/chest_xray/"
GENERATED_DATA_PATH = "./data/Data/covid-generated"
DATA_PATH = "./data"

# Assert that the data directories are present
for d in [COVID19_DATA_PATH, COVID19_AR_DATA_PATH, PNEUMONIA_DATA_PATH, DATA_PATH]:
    try:
        assert os.path.isdir(d) 
    except:
        print ("Directory %s does not exists" % d)

def create_list (split, use_generated=False):
    

    assert split in ['train', 'test', 'val']
    l = []

    # add generated images
    if ((use_generated==True) and (split=='train')):
        for f in glob.glob(os.path.join(GENERATED_DATA_PATH, 'genxray_*')):
            f = f.replace("\\", "/")
            l.append((f, 3)) # Class 0

    # Prepare list using kaggle pneumonia dataset
    for f in glob.glob(os.path.join(PNEUMONIA_DATA_PATH, split, 'NORMAL', '*')):
        f = f.replace("\\", "/")
        l.append((f, 0)) # Class 0

    for f in glob.glob(os.path.join(PNEUMONIA_DATA_PATH, split, 'PNEUMONIA', '*')):
        f = f.replace("\\", "/")
        if args.combine_pneumonia:
            l.append((f, 1)) # Class 1
        else:
            if 'bacteria' in f:
                l.append((f, 1)) # Class 1
            else:
                l.append((f, 2)) # Class 2

    # Prepare list using covid dataset
    covid_file = os.path.join(COVID19_DATA_PATH, '%s_list.txt'%split)
    with open(covid_file, 'r') as cf:
        for f in cf.readlines():
            f = f.replace("\n", "")
            f = f.replace("\\", "/")
            if args.combine_pneumonia:
                l.append((f, 2)) # Class 2
            else:
                l.append((f, 3)) # Class 3

    # Prepare list using covid AR dataset
    
    covid_ar_file = os.path.join(COVID19_AR_DATA_PATH, '%s_list.txt'%split)
    with open(covid_ar_file, 'r') as cf:
        for f in cf.readlines():
            f = f.replace("\n", "")
            f = f.replace("\\", "/")
            if args.combine_pneumonia:
                l.append((f, 2)) # Class 2
            else:
                l.append((f, 3)) # Class 3

    if ((use_generated==True) and (split=='train')):
        write_file = 'train_generated.txt'
    else:
        write_file = '%s.txt'%split
    with open(os.path.join(DATA_PATH, write_file), 'w') as f:
        for item in l:
            f.write("%s %d\n" % item)

for split in ['train', 'test', 'val']:
    create_list(split, args.use_generated)
