from data_collection.data_collection import CollectData
from data_modeling.training import train_model
from data_modeling.testing import test_model
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
	prepareModel()


if __name__ == '__main__':
	main()