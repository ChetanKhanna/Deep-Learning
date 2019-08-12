# The idea is to write a Python script that takes in path to an image folder
# and then run some sanity check on the images inside the folder. The valid images
# are then converted to numpy array which can be exported and stored on the disk

# importing required packages
import sys
import os
import numpy as np
from PIL import Image
import cv2


if len(sys.argv) < 2:
	raise Exception('Path not found. Run python3 conver_and_export_as_numpy.py [dirpath]')
if not os.path.exists(sys.argv[1]):
	raise Exception('No such file or directory.')
if not os.path.isdir(sys.argv[1]):
	raise Exception('Path is not a directory.')
PATH = os.path.abspath(sys.argv[1])
# sanity check
count = 0
for file in os.listdir(PATH):
	try:
		image = Image.open(os.path.join(PATH, file))
		image.verify()
	except (IOError, SyntaxError) as e:
		print('Bad file detected, deleting', file)
		os.remove(os.path.join(PATH, file))
		count += 1
print(count, 'files deleted.')
# making a numpy array
dataset = []
labels = []
label = os.path.basename(PATH)
HEIGHT, WIDTH = 128, 128
for file in os.listdir(PATH):
	full_size_img = cv2.imread(os.path.join(PATH, file))
	try:
		dataset.append(cv2.resize(full_size_img, (WIDTH, HEIGHT),
					   interpolation=cv2.INTER_CUBIC))
		labels.append(str(label))
	except:
		print('Cannot parse into numpy array:', file)
		remove = input('Remove[y]/n?') # asking user wheahter to remove file
		if not remove == 'n' or remove == 'N':
			os.remove(os.path.join(PATH, file))
# converting list object to numpy array
dataset = np.array(dataset)
labels = np.array(labels)
# converting dataset to 2D array
dataset = dataset.reshape((dataset.shape[0], HEIGHT*WIDTH*3))
print(dataset.shape)
np.save(PATH, dataset)
dr = np.load(PATH)
print(dr.shape)
