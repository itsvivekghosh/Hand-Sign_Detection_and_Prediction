from data_collection.data_collection import CollectData
from data_modeling.training import train_model
from data_modeling.testing.test_model import testModel
from prediction.predict import predict_in_video
import os, numpy as np


class PredictClass(object):

	def __init__(self):
		self.train_dir = 'data/train/'
		self.test_dir = 'data/test/'
		self.model_dir = 'model/'
		self.output_dir = 'outputs/'
		self.current_directory = os.getcwd()


	def CollectImages(self):

		self.data = CollectData(self.current_directory)
		self.data.generateData()


	def prepareModel(self):

		# train_model.trainTheModel(train_dir, test_dir)
		testModel(self.test_dir, self.model_dir, self.output_dir)
		pass


def main():

	obj = PredictClass()

	# obj.CollectImages()
	obj.prepareModel()
	predict_in_video(obj.model_dir)
	


if __name__ == '__main__':
	main()