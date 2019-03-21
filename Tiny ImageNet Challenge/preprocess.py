from keras.callbacks import Callback
import os

class EpochCheckpoint(Callback):
	def __init__(self, outputPath, every=5, startAt=0):
		# call the parent constructor
		super(Callback, self).__init__()

		# store the base output path for the model, the number of
		# epochs that must pass before the model is serialized to
		# disk and the current epoch value
		self.outputPath = outputPath
		self.every = every
		self.intEpoch = startAt

	def on_epoch_end(self, epoch, logs={}):
		# check to see if the model should be serialized to disk
		if (self.intEpoch + 1) % self.every == 0:
			p = os.path.sep.join([self.outputPath,
				"epoch_{}.hdf5".format(self.intEpoch + 1)])
			self.model.save(p, overwrite=True)

		# increment the internal epoch counter
		self.intEpoch += 1


from keras.preprocessing.image import img_to_array

class ImageToArrayPreprocessor:
	def __init__(self, dataFormat=None):
		# store the image data format
		self.dataFormat = dataFormat

	def preprocess(self, image):
		# apply the Keras utility function that correctly rearranges
		# the dimensions of the image
		return img_to_array(image, data_format=self.dataFormat)


# import the necessary packages
import cv2

class MeanPreprocessor:
	def __init__(self, rMean, gMean, bMean):
		# store the Red, Green, and Blue channel averages across a
		# training set
		self.rMean = rMean
		self.gMean = gMean
		self.bMean = bMean

	def preprocess(self, image):
		# split the image into its respective Red, Green, and Blue
		# channels
		(B, G, R) = cv2.split(image.astype("float32"))

		# subtract the means for each channel
		R -= self.rMean
		G -= self.gMean
		B -= self.bMean

		# merge the channels back together and return the image
		return cv2.merge([B, G, R])

# import the necessary packages
import cv2

class SimplePreprocessor:
	def __init__(self, width, height, inter=cv2.INTER_AREA):
		# store the target image width, height, and interpolation
		# method used when resizing
		self.width = width
		self.height = height
		self.inter = inter

	def preprocess(self, image):
		# resize the image to a fixed size, ignoring the aspect
		# ratio
		return cv2.resize(image, (self.width, self.height),
			interpolation=self.inter)