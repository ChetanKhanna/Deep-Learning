# importing required packages
import os
import sys
import numpy as np
import cv2
from PIL import Image
from google_images_download import google_images_download


# getting target image(s) from command line
if len(sys.argv) < 2:
	raise Exception('Run: python3 download_images [keyword] [number=100]')
elif len(sys.argv) == 2:
	KEYWORD, NUM_IMAGES = sys.argv[1], 100 # setting defualt to 100
else:
	if not sys.argv[-1].isdigit():
		KEYWORD, NUM_IMAGES = ' '.join(sys.argv[1:]), 100
	else:
		KEYWORD, NUM_IMAGES = ' '.join(sys.argv[1:-1]), sys.argv[-1]
# setting up directory and folder
BASE_DIR = os.path.expanduser('~')
DATA_DIR = os.path.join('Deep-Learning', 'data', 'Google Images')
DST_FOLDER = os.path.join(BASE_DIR, DATA_DIR)
# instantiating google_images_download class
response = google_images_download.googleimagesdownload()
arguments = {'keywords': KEYWORD, 'size': 'medium', 'limit': NUM_IMAGES,
			 'color_type': 'full-color', 'type': 'photo',
			 'aspect_ration': 'square', 'output_directory': DST_FOLDER}
paths = response.download(arguments)
# sanity checking: removing faulty images
count = 0
IMG_FOLDER = os.path.join(DST_FOLDER, KEYWORD)
for file in os.listdir(IMG_FOLDER):
	try:
		img = Image.open(os.path.join(IMG_FOLDER, file))
		img.verify()
	except (IOError, SyntaxError) as e:
		print('Bad file detected:', file)
		os.remove(os.path.join(IMG_FOLDER, file))
		count += 1
print('Bad files removed:', count)
# making a numpy dataset
dataset = []
labels = []
WIDTH, HEIGHT = 256, 256
for file in os.listdir(os.path.join(IMG_FOLDER)):
	full_size_img = cv2.imread(os.path.join(IMG_FOLDER, file))
	try:
		dataset.append(cv2.resize(full_size_img, (WIDTH, HEIGHT),
					   interpolation=cv2.INTER_CUBIC))
		labels.append('Thumbs-up')
	except:
		print('Cannot parse into numpy array:', file)
		remove = input('Remove[y]/n?') # asking user wheahter to remove file
		if not remove == 'n' or remove == 'N':
			os.remove(os.path.join(IMG_FOLDER, file))
# converting list object to numpy array
dataset = np.array(dataset)
labels = np.array(labels)
# converting dataset to 2D array
dataset = dataset.reshape((dataset.shape[0], HEIGHT*WIDTH*3))
print(dataset.shape)
