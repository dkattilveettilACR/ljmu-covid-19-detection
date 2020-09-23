import glob
import csv
import pandas as pd

if __name__ == '__main__':
	root_dir = './data/Data/covid-chestxray-dataset/images/'
	# field names  
	fields = ['filename', 'finding', 'view', 'modality']  
	filename = './data/covid_chestxray_dataset_reader/covid-chest-xray-data.csv'
	source_csv_file = './data/Data/covid-chestxray-dataset/metadata.csv'

	covid_data = pd.read_csv(source_csv_file, names = fields)
	covid_data.info()
	covid_data = covid_data[covid_data['modality']== 'X-ray'] #and covid_data['view'] != 'L'
	covid_data.info()
	#update finding to normal
	covid_data.loc[covid_data['finding'] == 'No Finding' , 'finding'] = 'normal'
	# update finding to viral pneumonia
	viral_pneumonia = ['Herpes pneumonia', 'Herpes pneumonia', 'ARDS', 'Influenza', 'MERS-CoV', 'SARS', 'Swine-Origin Influenza A (H1N1) Viral Pneumonia', 'Varicella']
	covid_data.loc[covid_data['finding'].isin(viral_pneumonia) , 'finding'] = 'viral pneumonia'
	# update finding to bacterial pneumonia
	bacterial_pneumonia = ['Bacterial', 'Chlamydophila', 'E.Coli', 'Klebsiella', 'Legionella', 'Lobar Pneumonia', 'Multilobar Pneumonia', 'Mycoplasma Bacterial Pneumonia', 'Round pneumonia', 'Streptococcus']
	covid_data.loc[covid_data['finding'].isin(bacterial_pneumonia) , 'finding'] = 'bacterial pneumonia'
	# update finding to covid pneumonia
	covid_pneumonia = ['COVID-19', 'COVID-19, ARDS']
	covid_data.loc[covid_data['finding'].isin(covid_pneumonia) , 'finding'] = 'covid_pneumonia'
	#writing to csv file  
	covid_data.to_csv(filename)
