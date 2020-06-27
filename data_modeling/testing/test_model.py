import tensorflow as tf
import keras
import os, sys
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import numpy as np
from keras.preprocessing.image import array_to_img, img_to_array
from keras.preprocessing import image
from matplotlib import rcParams


class TestImage(object):

	def __init__(self):
		self.image_path = ''
		self.model_path = '../../model/'
		self.test_dir, self.output_dir = '', ''
		self.image_size = (32, 32)
		self.model = None
		self.categories = {
			0: "ZERO",
			1: "ONE",
			2: "TWO",
			3: "THREE",
			4: "FOUR",
			5: "FIVE"
		}
		self.answer = None


	def __del__(self):
		cv2.waitKey(0)
		cv2.destroyAllWindows()


	def load_model_from_file(self):
		json_file = open(self.model_path+'model.json', 'r')
		loaded_model_json = json_file.read()
		json_file.close()

		loaded_model = keras.models.model_from_json(loaded_model_json)
		loaded_model.load_weights(self.model_path + "model_weights.h5")
		self.model = loaded_model
		print("Model Loaded!...")


	def preprocessImage(self, image):

		print("Processing Image!...")
		resized_image = cv2.resize(image, self.image_size)
		return resized_image


	def convertImageToArray(self, image):
		array = img_to_array(image)
		array = np.expand_dims(array, axis=0)
		return array


	def getRandomValue(self, rangeX, rangeY):
		
		import random
		# print(rangeX, rangeY)
		result = random.randint(rangeX, rangeY)
		return result


	def testImage(self, image_path = ''):

		for i in range(6):

			no_of_images_in_I = len(os.listdir(self.test_dir+"{}/".format(i)))
			random_image_value = self.getRandomValue(0, no_of_images_in_I+1)

			image = cv2.imread(self.test_dir+"{}/{}.jpg".format(i, random_image_value), 0)

			print("The Image is {}".format(self.test_dir+"{}/{}.jpg".format(i, random_image_value)))

			processed_image = self.preprocessImage(image)
			array = self.convertImageToArray(processed_image)

			self.load_model_from_file()
			self.classes = np.array(self.model.predict(array))[0]
			self.answer = self.categories[self.classes.argmax()]

			print("It's a: {}".format(self.answer))

			plt.imshow(image)
			plt.text(x=160, y=25, s=self.answer, fontsize=15)
			plt.savefig(self.output_dir+"{}.png".format(self.answer))
			plt.show()


def testModel(test_dir = '../../data/test/', model_path = '../../model/', output_dir = '../../outputs/'):

	rcParams['image.cmap'] = 'viridis'
	obj = TestImage()
	obj.test_dir = test_dir
	obj.model_path = model_path
	obj.output_dir = output_dir
	obj.testImage()
	print("Thanks!...")
	print("Exiting!...")
		

def main():

	testModel()


if __name__ == "__main__":

	print("Using Tensorflow Version: {}".format(tf.__version__))
	main()