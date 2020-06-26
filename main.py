from data_collection.data_collection import CollectData
import os


def main():
	current_directory = os.getcwd()
	data = CollectData(current_directory)
	data.generateData()

if __name__ == '__main__':
	main()