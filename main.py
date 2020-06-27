from data_collection.data_collection import CollectData
from data_modeling.training import train_model
from data_modeling.testing import test_model
from prediction.predict import predict_in_video
import os


def CollectImages():

	current_directory = os.getcwd()
	data = CollectData(current_directory)
	data.generateData()


def prepareModel():

	train_dir = 'Data/train/'
	test_dir = 'Data/test/'
	train_model.trainTheModel(train_dir, test_dir)
	test_model.testModel()
	pass


def main():

	# CollectImages()
	# prepareModel()
	predict_in_video('Model/')
	


if __name__ == '__main__':
	main()