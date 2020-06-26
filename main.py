from data_collection.data_collection import CollectData
from DataModeling.Training import train_model
import os


def CollectImages():

	current_directory = os.getcwd()
	data = CollectData(current_directory)
	data.generateData()


def prepareModel():

	train_model.main()
	pass


def main():

	CollectImages()
	prepareModel()


if __name__ == '__main__':
	main()