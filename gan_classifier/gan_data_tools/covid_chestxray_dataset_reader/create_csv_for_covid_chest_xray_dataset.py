import glob
import csv
import pandas as pd

if __name__ == '__main__':
	root_dir = './data/Data/covid-chest-xray-dataset/images/'
	# field names  
	fields = ['filename', 'finding', 'view', 'modality']  
	filename = './gan_classifier/gan_data_tools/covid_chestxray_dataset_reader/covid-chest-xray-data.csv'
	source_csv_file = './data/Data/covid-chest-xray-dataset/metadata.csv'

	covid_data = pd.read_csv(source_csv_file, usecols = fields)
	covid_data.head(5)
	
	covid_data = covid_data[(covid_data['modality']== 'X-ray') & (covid_data['view'] != 'L')]
	covid_data['finding'] = covid_data['finding'].str.upper()
	covid_data.info()
	
	#update finding to normal
	covid_data.loc[covid_data['finding'] == 'NO FINDING' , 'finding'] = 'normal'
	
	# update finding to viral pneumonia
	viral_pneumonia = ['HERPES PNEUMONIA', 'HERPES PNEUMONIA', 'ARDS', 'INFLUENZA', 'MERS-COV', 
					'SARS', 'SWINE-ORIGIN INFLUENZA A (H1N1) VIRAL PNEUMONIA', 'VARICELLA', 'HERPES PNEUMONIA, ARDS']
	covid_data.loc[covid_data['finding'].isin(viral_pneumonia) , 'finding'] = 'viral pneumonia'
	
	# update finding to bacterial pneumonia
	bacterial_pneumonia = ['BACTERIAL', 'CHLAMYDOPHILA', 'E.COLI', 'KLEBSIELLA', 'LEGIONELLA', 
						'LOBAR PNEUMONIA', 'MULTILOBAR PNEUMONIA', 'MYCOPLASMA BACTERIAL PNEUMONIA',
						'ROUND PNEUMONIA', 'STREPTOCOCCUS', 'MRSA']
	covid_data.loc[covid_data['finding'].isin(bacterial_pneumonia) , 'finding'] = 'bacterial pneumonia'
	
	# update finding to covid pneumonia
	covid_pneumonia = ['COVID-19', 'COVID-19, ARDS']
	covid_data.loc[covid_data['finding'].isin(covid_pneumonia) , 'finding'] = 'covid pneumonia'

	covid_data = covid_data.loc[covid_data['finding'].isin(['normal', 'viral pneumonia', 'bacterial pneumonia', 'covid pneumonia']), ['filename', 'finding']]
	covid_data['filename'] = root_dir + covid_data['filename']
	#writing to csv file  
	covid_data.to_csv(filename, index=False)
