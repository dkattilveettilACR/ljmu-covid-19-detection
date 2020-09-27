import glob
import csv
import os
import pydicom
import numpy as np
import cv2
from PIL import Image
import ntpath
import png

if __name__ == '__main__':
	root_dir = './data/Data/covid-19-AR/'
	image_dir = './data./Data./COVID-19-AR-images/'
	# field names  
	fields = ['Image Index', 'Finding Labels']  
	source_csv_file = './data/covid-ar-data-reader/covid-ar-data.csv'
	# writing to csv file  
	with open(source_csv_file, 'w', newline='') as csvfile:  
		# creating a csv writer object  
		csvwriter = csv.writer(csvfile)  
		csvwriter.writerow(fields)  
		# read DICOM files in the folder and create PNG images
		count = 0
		for filename in glob.iglob(root_dir + '**/*.dcm', recursive=True):
			
			dcm_file = pydicom.dcmread(filename)
			modality = dcm_file[0x08,0x60].value
			patient_id = dcm_file[0x10,0x20].value
			study_description = dcm_file[0x08,0x1030].value
			
			print("modality:"+ modality + ", study description:" + study_description )
			
			if ((modality in ['CT', 'CR']) | ('LATERAL' in study_description)):
				continue

			image_2d = dcm_file.pixel_array.astype(float)
			shape = dcm_file.pixel_array.shape
			# Rescaling grey scale between 0-255
			image_2d_scaled = (np.maximum(image_2d,0) / image_2d.max()) * 255.0

			# Convert to uint
			image_2d_scaled = np.uint8(image_2d_scaled)

			# Write the PNG file
			count += 1
			image_file = image_dir + str(count) +  "_"  + patient_id + "_" + modality + "_" + ntpath.basename(filename).replace('.dcm','.png')
			with open(image_file, 'wb') as png_file:
				w = png.Writer(shape[1], shape[0], greyscale=True)
				w.write(png_file, image_2d_scaled)

			finding = 'covid pneumonia'
			csvwriter.writerow([image_file, finding])
			