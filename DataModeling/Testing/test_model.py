import tensorflow as tf
import keras
import os, sys
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import numpy as np
from keras.preprocessing.image import array_to_img, array_to_img

class TestImage(object):

	def __init__(self):
		self.image_path = ''
		self.model_path = '../../Model/'
		self.test_path = '../../Data/test/'
		self.image_size = (32, 32)
		pass

	def __del__(self):
		cv2.waitKey(0)
		cv2.destroyAllWindows()
		pass


	def preprocessImage(self, image):
		pass

	def testImage(self, image_path = ''):
		image = cv2.imread(self.test_path+"0/1.jpg")
		cv2.imshow("Image", image)
		pass

		
def main():

	print("Using Tensorflow Version: {}".format(tf.__version__))
	obj = TestImage()
	obj.testImage()

if __name__ == "__main__":
	main()