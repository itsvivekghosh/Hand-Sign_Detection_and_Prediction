import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Input, Flatten
from keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img
import keras
import os

class PrepareModel(object):

	def __init__(self, train_dir, test_dir):
		self.model = Sequential()
		self.datagen = None
		self.loss = keras.losses.categorical_crossentropy
		self.optimizer = keras.optimizers.Adam(lr=0.001)
		self.metrics = ['accuracy']

		self.train_dir = train_dir
		self.test_dir = test_dir


	def prepareImageGenerator(self):

		train_datagen = ImageDataGenerator(
			vertical_flip = True, 
			horizontal_flip = True, 
			shear_range = 0.3, 
			zoom_range = 0.2,
			rescale = 1. / 255.
		)

		test_datagen = ImageDataGenerator(rescale = 1./255.)

		train_set = train_datagen.flow_from_directory(
			self.train_dir, 
			target_size = (200, 200),
			batch_size=100,
			color_mode = 'grayscale',
			class_mode = 'categorical'
			)

		train_set = train_datagen.flow_from_directory(
			self.test_dir, 
			target_size = (200, 200),
			batch_size=100,
			color_mode = 'grayscale',
			class_mode = 'categorical'
			)

		return train_set, test_set


	def setModel(self):
		
		self.model.add(Conv2D(filters=256, kernel_size = (3, 3), activation = 'relu', input_shape = (200, 200, 1)))
		self.model.add(Conv2D(filters=256, kernel_size = (3, 3), activation = 'relu'))
		self.model.add(MaxPooling2D(pool_size=(2, 2)))
		self.model.add(Dropout(0.25))

		self.model.add(Conv2D(filters=512, kernel_size = (3, 3), activation = 'relu', input_shape = (200, 200, 1)))
		self.model.add(Conv2D(filters=512, kernel_size = (3, 3), activation = 'relu'))
		self.model.add(MaxPooling2D(pool_size=(2, 2)))
		self.model.add(Dropout(0.25))

		self.model.add(Flatten())
		self.model.add(Dense(1024, activation='relu'))
		self.model.add(Dropout(0.25))
		self.model.add(BatchNormalization())
		self.model.add(Dense(6, activation='softmax'))

		self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)
		print(self.model.summary())


	def trainModel(self):

		train_set, test_set = self.prepareImageGenerator()

		self.model.fit_generator(
			train_set, 
			steps_per_epoch = 500,
			epochs= 10,
			validation_data = test_set,
			validation_steps = len(test_set)
			)

def main():

	train_dir, test_dir = '../../Data/train/', '../../Data/test'

	prep = PrepareModel(train_dir, test_dir)
	prep.setModel()


if __name__ == '__main__':
	main()