import pandas as pd
import csv	
import os.path
from os import path

if __name__ == '__main__':
    chest_xray_data_file = './gan_classifier/gan_data_tools/chest_xray_data_reader/chest-xray-data.csv'
    covid_chest_xray_data_file = './gan_classifier/gan_data_tools/covid_chestxray_dataset_reader/covid-chest-xray-data.csv'
    covid_ar_data_file = './gan_classifier/gan_data_tools/covid-ar-data-reader/covid-ar-data.csv'

    #merge data from 3 files
    fields = ['filename', 'finding']  
    chest_xray_data = pd.read_csv(chest_xray_data_file, usecols = fields)
    covid_chest_xray_data = pd.read_csv(covid_chest_xray_data_file, usecols = fields)
    covid_ar_data = pd.read_csv(covid_ar_data_file, usecols = fields)

    dataTrain = pd.concat([chest_xray_data, covid_chest_xray_data, covid_ar_data], axis = 0)
    csv_file = './gan_classifier/gan_data_tools/metadata.csv'
    
    class_dict = {'normal':0, 'bacterial pneumonia':1, 'viral pneumonia':2, 'covid pneumonia':3}
    with open(csv_file, 'w', newline='') as csvfile:   
        csvwriter = csv.writer(csvfile)  
        csvwriter.writerow(fields)  
        for index, row in dataTrain.iterrows():
            if (path.exists(row["filename"])):
                csvwriter.writerow([row["filename"], class_dict[row["finding"]]])