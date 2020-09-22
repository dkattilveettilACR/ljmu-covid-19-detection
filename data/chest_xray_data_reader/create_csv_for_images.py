import glob
import csv

if __name__ == '__main__':
	root_dir = './data/Data/chest-xray-pneumonia/chest_xray/'
	# field names  
	fields = ['Image Index', 'Finding Labels']  
	filename = './data/chest_xray_data_reader/chest-xray-data.csv'
	# writing to csv file  
	with open(filename, 'w', newline='') as csvfile:  
		# creating a csv writer object  
		csvwriter = csv.writer(csvfile)  
		csvwriter.writerow(fields)  
		for filename in glob.iglob(root_dir + '**/*.jpeg', recursive=True):
			finding = 'bacterial pneumonia'
			filename_upper = filename.upper()
			if 'NORMAL' in filename_upper:
				finding = 'normal'
			elif 'VIRUS' in filename_upper:
				finding = 'viral pneumonia'
			csvwriter.writerow([filename, finding])