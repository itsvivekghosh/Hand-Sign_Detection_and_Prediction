import tensorflow as tf
import numpy as np
import keras
from PIL import Image
import os, cv2, sys, PIL
from keras.preprocessing import image



class Predict(object):

	def __init__(self):
		pass



def predict():
	obj = Predict()
	pass

def main():
	predict()

if __name__ == "__main__":
	print("Using Tensorflow version: {}".format(tf.__version__))
	print("Using Keras version: {}".format(keras.__version__))
	print("Using Pillow version: {}".format(PIL.__version__))
	print("Using Numpy version: {}".format(np.__version__))
	print("Using cv2 version: {}".format(cv2.__version__))
	
	main()