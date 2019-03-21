import os
import cv2
import imutils
import json
import imutils
import numpy as np
from dataset_writer import HDF5DatasetWriter
from sklearn.preprocessing import LabelEncoder
from imutils import paths

TRAIN = "../tiny-imagenet-200/train/"
VAL = "../tiny-imagenet-200/val/images"
VAL_ANNOT = "../tiny-imagenet-200/val/val_annotations.txt"
WORDNET = "../tiny-imagenet-200/wnits.txt"
WORD_LABELS = "../tiny-imagenet-200/words.txt"

CLASSES = 200
UM_IMAGES = 500 * CLASSES

TRAIN_HF5 = "../tiny-imagenet-200/hf5/train.hdf5"
VAL_HF5 = "../tiny-imagenet-200/hf5/val.hdf5"

MEAN_NORM = "tiny-image-net-200-mean.json"
OUTPUT = "output"

le = LabelEncoder() 

# Train set configurations
train_paths = list(paths.list_images(TRAIN))
train_labels = [x.split(os.path.sep)[-3] for x in train_paths]
train_labels = le.fit_transform(train_labels)

# Validation set configurations
temp = open(VAL_ANNOT).read().strip().split("\n")
temp = [x.split("\t")[:2] for x in temp]
val_paths = [os.path.sep.join([VAL, x[0]]) for x in temp]
val_labels = [x[1] for x in temp]
val_labels = le.fit_transform(val_labels)

datasets = [
	("train", train_paths, train_labels, TRAIN_HF5),
	("val", val_paths, val_labels, VAL_HF5)
]

(R, G, B) = ([], [], [])

for (dType, paths, labels, outputPath) in datasets:
	writer = HDF5DatasetWriter((len(paths), 64, 64, 3), outputPath)

	for (i, (path, label)) in enumerate(zip(paths, labels)):
		image = cv2.imread(path)

		# Storing mean averages of train set data
		if dType == "train":
			(b, g, r) = cv2.mean(image)[:3]
			R.append(r)
			G.append(g)
			B.append(b)
		
		# Add the image and label to the HDF5 dataset
		writer.add([image], [label])

	writer.close()

means = {"R": np.mean(R), "G": np.mean(G), "B": np.mean(B)}
file = open(MEAN_NORM, "w")
file.write(json.dumps(means))
file.close()