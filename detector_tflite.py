import cv2
import numpy as np
#import tensorflow as tf
#from object_detection.utils import label_map_util
import importlib.util


class DetectorTF2:

	def __init__(self, path_to_checkpoint, path_to_labelmap, class_id=None, threshold=0.5):

		# edge TPU is essential for now
		use_TPU = True
		self.Threshold = threshold
		
		# Load the label map
		with open(path_to_labelmap, 'r') as f:
			self.labels = [line.strip() for line in f.readlines()]

		# Import TensorFlow libraries
		# If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
		# If using Coral Edge TPU, import the load_delegate library
		pkg = importlib.util.find_spec('tflite_runtime')
		if pkg:
			from tflite_runtime.interpreter import Interpreter
			if use_TPU:
					from tflite_runtime.interpreter import load_delegate
		else:
			from tensorflow.lite.python.interpreter import Interpreter
			if use_TPU:
					from tensorflow.lite.python.interpreter import load_delegate

		#CWD_PATH = os.getcwd()
		#GRAPH_NAME = 'model_full_integer_quant_edgetpu.tflite'
		#PATH_TO_CKPT = os.path.join(CWD_PATH, args.model_path, GRAPH_NAME)
		PATH_TO_CKPT = path_to_checkpoint
		# Load the Tensorflow Lite model.
		# If using Edge TPU, use special load_delegate argument
		if use_TPU:
			self.interpreter = Interpreter(model_path=PATH_TO_CKPT, experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
			print(PATH_TO_CKPT)
		else:
			self.interpreter = Interpreter(model_path=PATH_TO_CKPT)

		self.interpreter.allocate_tensors()

		# Get model details
		self.input_details = self.interpreter.get_input_details()
		self.output_details = self.interpreter.get_output_details()
		self.height = self.input_details[0]['shape'][1]
		self.width = self.input_details[0]['shape'][2]
		
		#self.height = 640
		#self.width = 640
		
		input_mean = 127.5
		input_std = 127.5


	def DetectFromImage(self, img):
		#im_height, im_width, _ = img.shape
		self.imH, self.imW, _ = img.shape
		img = cv2.resize(img, (self.width, self.height))
		# Expand dimensions since the model expects images to have shape: [1, None, None, 3]
		input_tensor = np.expand_dims(img, 0)
		#detections = self.detect_fn(input_tensor)
		input_tensor = np.array(input_tensor - 127.5, dtype=np.int8)
		
		self.interpreter.set_tensor(self.input_details[0]['index'],input_tensor)
		self.interpreter.invoke()

		boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0] # Bounding box coordinates of detected objects
		self.classes = self.interpreter.get_tensor(self.output_details[1]['index'])[0] # Class index of detected objects
		self.scores = self.interpreter.get_tensor(self.output_details[2]['index'])[0] # Confidence of detected objects

		self.scores = (self.scores + 128) / 255
		self.scores = (self.scores[0],)
		boxes = (boxes + 85.1) / 160

		#bboxes = detections['detection_boxes'][0].numpy()
		#bclasses = detections['detection_classes'][0].numpy().astype(np.int32)
		#bscores = detections['detection_scores'][0].numpy()
		det_boxes = self.ExtractBBoxes(boxes, self.classes, self.scores, self.imW, self.imH)
		return det_boxes


	def ExtractBBoxes(self, boxes, classes, scores, imW, imH):
		bbox = []
		for i in range(len(scores)):
			if ((scores[i] > self.Threshold) and (scores[i] <= 1.0)):
            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
				ymin = int(max(1,(boxes[i][0] * self.imH)))
				xmin = int(max(1,(boxes[i][1] * self.imW)))
				ymax = int(min(self.imH,(boxes[i][2] * self.imH)))
				xmax = int(min(self.imW,(boxes[i][3] * self.imW)))
				#print(i,xmin,ymin,xmax,ymax)
				object_name = self.labels[int(classes[i])] # Look up object name from "labels" array using class index
				class_label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
				bbox.append([xmin, ymin, xmax, ymax, class_label, float(scores[i])])
		return bbox


	def DisplayDetections(self, image, boxes_list, det_time=None):
		if not boxes_list: return image  # input list is empty
		img = image.copy()
		for idx in range(len(boxes_list)):
			x_min = boxes_list[idx][0]
			y_min = boxes_list[idx][1]
			x_max = boxes_list[idx][2]
			y_max = boxes_list[idx][3]
			cls =  str(boxes_list[idx][4])
			score = str(np.round(boxes_list[idx][-1], 2))

			#text = cls + ": " + score
			text = cls
			cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
			cv2.rectangle(img, (x_min, y_min - 20), (x_min, y_min), (255, 255, 255), -1)
			cv2.putText(img, text, (x_min + 5, y_min - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

		if det_time != None:
			fps = round(1000. / det_time, 1)
			fps_txt = str(fps) + " FPS"
			cv2.putText(img, fps_txt, (25, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

		return img

