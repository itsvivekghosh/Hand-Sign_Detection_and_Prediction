import cv2
import numpy as np
import os
import sys


class CollectData(object):

	def __init__(self, curr_dir):

		print("Initializing...")
		self.curr_dir = curr_dir
		self.model_train_path = '../Data/train/'
		self.model_test_path = '../Data/test/'

		if not os.path.exists("Data"):

			os.makedirs(self.model_train_path)
			os.makedirs(self.model_test_path)

			for i in range(6):
				os.makedirs(self.model_train_path+'{}'.format(i))
				os.makedirs(self.model_test_path+"{}".format(i))

		self.mode = 'train'
		self.working_dir = 'Data/' + self.mode

		self.images_count = {}

		self.cap = cv2.VideoCapture(0)


	def __del__(self):

		self.cap.release()
		cv2.destroyAllWindows()
		print("Exiting...")

	def generateTextIntoFrame(self, frame):
		
		cv2.putText(frame, "MODE: " + self.mode, (10, 50), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
		cv2.putText(frame, "IMAGE COUNT: ", (10, 100), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
		cv2.putText(frame, "ZERO: " + str(self.images_count['zero']), (10, 120), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
		cv2.putText(frame, "ONE: " + str(self.images_count['one']), (10, 150), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
		cv2.putText(frame, "TWO: " + str(self.images_count['two']), (10, 180), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
		cv2.putText(frame, "THREE: " + str(self.images_count['three']), (10, 210), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
		cv2.putText(frame, "FOUR: " + str(self.images_count['four']), (10, 240), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
		cv2.putText(frame, "FIVE: " + str(self.images_count['five']), (10, 270), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)


	def drawROI(self, frame):

		# Coordinates of the ROI
	    x1 = int(0.5 * frame.shape[1])
	    y1 = 10
	    x2 = frame.shape[1]-10
	    y2 = int(0.5 * frame.shape[1])
	    # Drawing the ROI
	    # The increment/decrement by 1 is to compensate for the bounding box
	    cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (0, 255 , 0) ,1)
	    # Extracting the ROI
	    roi = frame[y1:y2, x1:x2]
	    roi = cv2.resize(roi, (200, 200)) 

	    return roi


	def countImages(self):

		self.images_count = {
				'zero': len(os.listdir(self.working_dir+'/0')),
				'one': len(os.listdir(self.working_dir+'/1')),
				'two': len(os.listdir(self.working_dir+'/2')),
				'three': len(os.listdir(self.working_dir+'/3')),
				'four': len(os.listdir(self.working_dir+'/4')),
				'five': len(os.listdir(self.working_dir+'/5'))
			}

	def generateData(self):
		
		while True:

			self.countImages()

			res, frame = self.cap.read()

			frame = cv2.flip(frame, 1)

			self.generateTextIntoFrame(frame)
			roi = self.drawROI(frame)

			roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
			_, roi = cv2.threshold(roi, 122, 255, cv2.THRESH_BINARY)

			cv2.imshow("Frame", frame)
			cv2.imshow("ROI", roi)

			key = cv2.waitKey(10)

			if key == 27 or key == ord('q'):
				break

			if key == ord('0'):
				name_dir = str( self.images_count['zero'] ) + '.jpg'
				directory = self.working_dir + '/{}/{}'.format(0, name_dir)
				cv2.imwrite(directory, roi)
				print('Saved for {} as {} in {} Mode'.format("zero", name_dir, self.mode))
				pass

			if key == ord('1'):
				name_dir = str( self.images_count['one'] ) + '.jpg'
				directory = self.working_dir + '/{}/{}'.format(1, name_dir)
				cv2.imwrite(directory, roi)
				print('Saved for {} as {} in {} Mode'.format("one", name_dir, self.mode))
				pass

			if key == ord('2'):
				name_dir = str( self.images_count['two'] ) + '.jpg'
				directory = self.working_dir + '/{}/{}'.format(2, name_dir)
				cv2.imwrite(directory, roi)
				print('Saved for {} as {} in {} Mode'.format("two", name_dir, self.mode))
				pass

			if key == ord('3'):
				name_dir = str( self.images_count['three'] ) + '.jpg'
				directory = self.working_dir + '/{}/{}'.format(3, name_dir)
				cv2.imwrite(directory, roi)
				print('Saved for {} as {} in {} Mode'.format("three", name_dir, self.mode))
				pass

			if key == ord('4'):
				name_dir = str( self.images_count['four'] ) + '.jpg'
				directory = self.working_dir + '/{}/{}'.format(4, name_dir)
				cv2.imwrite(directory, roi)
				print('Saved for {} as {} in {} Mode'.format('four', name_dir, self.mode))
				pass

			if key == ord('5'):
				name_dir = str( self.images_count['five'] ) + '.jpg'
				directory = self.working_dir + '/{}/{}'.format(5, name_dir)
				cv2.imwrite(directory, roi)
				print('Saved for {} as {} in {} Mode'.format('five', name_dir, self.mode))
				pass

			if key == ord('c'):
				if self.mode == 'train':
					self.mode = 'test'
				else:
					self.mode = 'train'
				self.working_dir = 'Data/'+self.mode	


def main():

	current_directory = os.getcwd()
	obj = CollectData(current_directory)
	obj.generateData()


if __name__ == '__main__':
	main()