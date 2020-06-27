import tensorflow as tf
import numpy as np
import keras
from PIL import Image
import os, cv2, sys, PIL, json
from keras.preprocessing import image



class Predict(object):

	def __init__(self):
		self.cap = cv2.VideoCapture(0)
		self.model_path = ''
		self.model = None
		self.image_size = (32, 32)
		self.categories = {
			0: "ZERO",
			1: "ONE",
			2: "TWO",
			3: "THREE",
			4: "FOUR",
			5: "FIVE"
		}
		self.answer = -1
		self.flag = False
		pass


	def __del__(self):
		self.cap.release()
		cv2.destroyAllWindows()

		print("Exiting!...")


	def loadModel(self, model_path):

		self.model_path = model_path
		json_file = open(self.model_path + 'model.json', 'r')
		loaded_model_json = json_file.read()
		json_file.close()

		loaded_model = keras.models.model_from_json(loaded_model_json)
		loaded_model.load_weights(self.model_path + 'model_weights.h5')
		self.model = loaded_model

		print("Model Loaded!...")



	def drawROI(self, frame):

		# Coordinates of the ROI
	    x1 = int(0.5 * frame.shape[1])
	    y1 = 10
	    x2 = frame.shape[1]-10
	    y2 = int(0.5 * frame.shape[1])
	    # Drawing the ROI
	    # The increment/decrement by 1 is to compensate for the bounding box
	    cv2.rectangle(frame, (x1+40, y1+40), (x2-10, y2-10), (0, 255 , 0) ,1)
	    # Extracting the ROI
	    roi = frame[y1:y2, x1:x2]
	    roi = cv2.resize(roi, (32, 32))

	    return x1+40, y1+40, roi


	def preprocessImage(self, frame):
		resized_image = cv2.resize(frame, self.image_size)
		return resized_image


	def convertImageToArray(self, frame):
		array = image.img_to_array(frame)
		array = np.expand_dims(array, axis=0)
		return array


	def predictImage(self, model_path = '../../model/'):

		self.loadModel(model_path)

		while True:
			res, frame = self.cap.read()

			if res == 0:
				return

			frame = cv2.flip(frame, 1)

			x, y, roi = self.drawROI(frame)
			roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
			_, roi = cv2.threshold(roi, 122, 255, cv2.THRESH_BINARY)

			### Preprocessing the Image
			preprocessed_image = self.preprocessImage(roi)
			array = self.convertImageToArray(preprocessed_image)

			### Prediction
			if self.flag == True:
				self.classes = np.array(self.model.predict(array))[0]
				# print(self.classes)
				self.answer = self.categories[self.classes.argmax()]
			
				cv2.putText(frame, self.answer, (x, y), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 0), 1)

			else:
				cv2.putText(frame, "NO ANSWER", (x, y), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 0), 1)

			cv2.imshow("Prediction", frame)
			cv2.imshow("ROI", roi)

			key = cv2.waitKey(1)

			if key == ord("p"):
				self.flag = not self.flag
				pass

			elif key == ord("q") or key == 27:
				break



def predict_in_video(model_path = '../model/'):
	obj = Predict()
	obj.predictImage(model_path)


def main():
	predict_in_video()


if __name__ == "__main__":
	print("Using Tensorflow version: {}".format(tf.__version__))
	print("Using Keras version: {}".format(keras.__version__))
	print("Using Pillow version: {}".format(PIL.__version__))
	print("Using Numpy version: {}".format(np.__version__))
	print("Using cv2 version: {}".format(cv2.__version__))

	main()