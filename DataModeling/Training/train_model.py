import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Input, Flatten, Activation
from keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import plot_model
from IPython.display import SVG, Image
from livelossplot import *
import tensorflow as tf
import keras
import os
import matplotlib.pyplot as plt


class PrepareModel(object):

	def __init__(self, train_dir, test_dir):
		
		self.model = Sequential()
		self.target_size = (32, 32)
		self.batch_size = 10
		self.datagen = None
		self.loss = keras.losses.categorical_crossentropy
		self.optimizer = keras.optimizers.Adam(lr=0.001)
		self.metrics = ['accuracy']
		self.history = None
		self.train_dir = train_dir
		self.test_dir = test_dir
		self.model_checkpoint = None
		self.reduce_lr = None
		self.callbacks = None
		self.model_path = '../../Model/'

	def prepareImageGenerator(self):

		train_datagen = ImageDataGenerator(
			horizontal_flip = True, 
			vertical_flip = True,
			shear_range = 0.3, 
			zoom_range = 0.2,
			rescale = 1/255.
		)

		test_datagen = ImageDataGenerator(
			rescale = 1/255.
		)

		train_set = train_datagen.flow_from_directory(
			self.train_dir, 
			target_size = self.target_size,
			batch_size=self.batch_size,
			color_mode = 'grayscale',
			class_mode = 'categorical',
			shuffle = True
		)

		test_set = train_datagen.flow_from_directory(
			self.test_dir, 
			target_size = self.target_size,
			batch_size=self.batch_size,
			color_mode = 'grayscale',
			class_mode = 'categorical',
			shuffle = True
		)

		return train_set, test_set


	def setModel(self):
		
		self.model.add(Conv2D(filters=32, kernel_size = (3, 3), padding='same', kernel_initializer = 'normal', input_shape = (32, 32, 1)))
		self.model.add(Activation("relu"))
		self.model.add(Conv2D(filters=32, kernel_size = (3, 3), padding='same', kernel_initializer = 'normal'))
		self.model.add(Activation("relu"))
		self.model.add(MaxPooling2D(pool_size=(2, 2)))
		self.model.add(Dropout(0.25))

		self.model.add(Conv2D(filters=64, kernel_size = (3, 3), padding='same', kernel_initializer = 'normal'))
		self.model.add(Activation("relu"))
		self.model.add(Conv2D(filters=64, kernel_size = (3, 3), padding='same', kernel_initializer = 'normal'))
		self.model.add(Activation("relu"))
		self.model.add(MaxPooling2D(pool_size=(2, 2)))
		self.model.add(Dropout(0.25))

		self.model.add(Conv2D(filters=128, kernel_size = (3, 3), padding='same', kernel_initializer = 'normal'))
		self.model.add(Activation("relu"))
		self.model.add(Conv2D(filters=128, kernel_size = (3, 3), padding='same', kernel_initializer = 'normal'))
		self.model.add(Activation("relu"))
		self.model.add(MaxPooling2D(pool_size=(2, 2)))
		self.model.add(Dropout(0.25))

		self.model.add(Flatten())
		self.model.add(Dense(256))
		self.model.add(Activation("relu"))
		self.model.add(Dropout(0.25))
		self.model.add(Dense(6, activation='softmax'))

		self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)
		self.model_checkpoint = ModelCheckpoint(self.model_path+"model_weights.h5", monitor='val_accuracy',
                            save_weights_only=True, model='max', verbose=5)

		self.reduce_lr = ReduceLROnPlateau(
		    monitor='val_loss', 
		    factor=0.1, 
		    patience=2, 
		    min_lr=0.01,
		    model='auto'
		)

		self.callbacks = [
		    PlotLossesKeras(), self.model_checkpoint, self.reduce_lr
		]

		print(self.model.summary())

	def saveModel(self):
		
		model_json = self.model.to_json()
		with open(self.model_path+"model.json", "w") as json_file:
		    json_file.write(model_json)
		print("Model Saved!")


	def trainModel(self):

		train_set, test_set = self.prepareImageGenerator()

		self.history = self.model.fit_generator(
			train_set, 
			steps_per_epoch = len(train_set),
			epochs= 10,
			validation_data = test_set,
			validation_steps = len(test_set),
			callbacks = self.callbacks
		)
		self.saveModel()



def trainTheModel(train_dir = '../../Data/train/', test_dir = '../../Data/test'):

	prep = PrepareModel(train_dir, test_dir)
	prep.setModel()
	prep.trainModel()


def main():
	trainTheModel()
	

if __name__ == '__main__':
	print("Tensorflow version:", tf.__version__)
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
	main()