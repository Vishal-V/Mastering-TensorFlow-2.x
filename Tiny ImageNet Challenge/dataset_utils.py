import h5py
import os

class HDF5DatasetWriter:
	def __init__(self, dims, outputPath, dataKey="images",
		bufSize=1000):
		# check to see if the output path exists, and if so, raise
		# an exception
		if os.path.exists(outputPath):
			raise ValueError("The supplied `outputPath` already "
				"exists and cannot be overwritten. Manually delete "
				"the file before continuing.", outputPath)

		# open the HDF5 database for writing and create two datasets:
		# one to store the images/features and another to store the
		# class labels
		self.db = h5py.File(outputPath, "w")
		self.data = self.db.create_dataset(dataKey, dims,
			dtype="float")
		self.labels = self.db.create_dataset("labels", (dims[0],),
			dtype="int")

		# store the buffer size, then initialize the buffer itself
		# along with the index into the datasets
		self.bufSize = bufSize
		self.buffer = {"data": [], "labels": []}
		self.idx = 0

	def add(self, rows, labels):
		# add the rows and labels to the buffer
		self.buffer["data"].extend(rows)
		self.buffer["labels"].extend(labels)

		# check to see if the buffer needs to be flushed to disk
		if len(self.buffer["data"]) >= self.bufSize:
			self.flush()

	def flush(self):
		# write the buffers to disk then reset the buffer
		i = self.idx + len(self.buffer["data"])
		self.data[self.idx:i] = self.buffer["data"]
		self.labels[self.idx:i] = self.buffer["labels"]
		self.idx = i
		self.buffer = {"data": [], "labels": []}

	def storeClassLabels(self, classLabels):
		# create a dataset to store the actual class label names,
		# then store the class labels
		dt = h5py.special_dtype(vlen=str) # `vlen=unicode` for Py2.7
		labelSet = self.db.create_dataset("label_names",
			(len(classLabels),), dtype=dt)
		labelSet[:] = classLabels

	def close(self):
		# check to see if there are any other entries in the buffer
		# that need to be flushed to disk
		if len(self.buffer["data"]) > 0:
			self.flush()

		# close the dataset
		self.db.close()


# import the necessary packages
from keras.utils import np_utils
import numpy as np
import h5py

class HDF5DatasetGenerator:
	def __init__(self, dbPath, batchSize, preprocessors=None,
		aug=None, binarize=True, classes=2):
		# store the batch size, preprocessors, and data augmentor,
		# whether or not the labels should be binarized, along with
		# the total number of classes
		self.batchSize = batchSize
		self.preprocessors = preprocessors
		self.aug = aug
		self.binarize = binarize
		self.classes = classes

		# open the HDF5 database for reading and determine the total
		# number of entries in the database
		self.db = h5py.File(dbPath)
		self.numImages = self.db["labels"].shape[0]

	def generator(self, passes=np.inf):
		# initialize the epoch count
		epochs = 0

		# keep looping infinitely -- the model will stop once we have
		# reach the desired number of epochs
		while epochs < passes:
			# loop over the HDF5 dataset
			for i in np.arange(0, self.numImages, self.batchSize):
				# extract the images and labels from the HDF dataset
				images = self.db["images"][i: i + self.batchSize]
				labels = self.db["labels"][i: i + self.batchSize]

				# check to see if the labels should be binarized
				if self.binarize:
					labels = np_utils.to_categorical(labels,
						self.classes)

				# check to see if our preprocessors are not None
				if self.preprocessors is not None:
					# initialize the list of processed images
					procImages = []

					# loop over the images
					for image in images:
						# loop over the preprocessors and apply each
						# to the image
						for p in self.preprocessors:
							image = p.preprocess(image)

						# update the list of processed images
						procImages.append(image)

					# update the images array to be the processed
					# images
					images = np.array(procImages)

				# if the data augmenator exists, apply it
				if self.aug is not None:
					(images, labels) = next(self.aug.flow(images,
						labels, batch_size=self.batchSize))

				# yield a tuple of images and labels
				yield (images, labels)

			# increment the total number of epochs
			epochs += 1

	def close(self):
		# close the database
		self.db.close()