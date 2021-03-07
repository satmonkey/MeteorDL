# by Milan Kalina
# based on Tensorflow API v2 Object Detection code
# https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2.md

from os import path, system, mkdir, makedirs
import cv2
from matplotlib import pyplot as plt
import time
import argparse
import numpy as	np
#import cupy as cp
from datetime import datetime
from threading import Thread, Semaphore, Lock
from detector import DetectorTF2
import dvg_ringbuffer as rb
from PIL import Image
#import cumpy

class VideoStreamWidget(object):
	def	__init__(self, src=0):
			# Create a VideoCapture object
			self.capture = cv2.VideoCapture(src)
			# Start the thread to read frames from the video stream
			self.thread	= Thread(target=self.update_rb, args=())
			self.thread.daemon = True
			self.thread.start()

	def	update_rb(self):
			# Read the next frame from the stream in a different thread
			# and maintains buffer - list of consequtive frames
			self.total = int(args.fps) * 5	# buffer size
			self.k = 0			# global frame counter
			self.j = 0			# maxpixel counter
			self.t = [0,0]	# time tracking
			self.h = [] 		# history frame index
			mutex = Lock()
			self.time0 = time.time()
			(self.status, self.frame) = self.capture.read()
			self.frame_width = self.frame.shape[1]
			self.frame_height = self.frame.shape[0]
			self.np_buffer = rb.RingBuffer(self.total, dtype=(np.uint8,(self.frame_height, self.frame_width, 3)), allow_overwrite=False)
			print("Filling the ring buffer...")
			self.buffer_fill()
			print("Buffer filled, starting...")
			while True:
				if self.capture.isOpened():
					(self.status, self.frame) = self.capture.read()
					if (self.status):
						cv2.rectangle(self.frame, (0, (self.frame_height-10)), (200, self.frame_height), (0,0,0), -1)
						cv2.putText(self.frame, station + ' ' + datetime.utcfromtimestamp(time.time()).strftime('%d/%m/%Y %H:%M:%S.%f')[:-4] + ' ' + str(self.k), (1,self.frame_height-3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1, cv2.LINE_AA)
						mutex.acquire()
						self.np_buffer.popleft()
						self.np_buffer.append(self.frame)
						mutex.release()
						self.t.append((self.k, time.time()))
						self.k += 1
						self.j += 1
					else:
						print("\n" + 'Connection lost... trying to restore...' + datetime.utcfromtimestamp(time.time()).strftime('%H:%M:%S'))
						self.con_restore()
						print('Connection restored...' + datetime.utcfromtimestamp(time.time()).strftime('%H:%M:%S'))
				else:
					print('Capture closed...')
					
	def buffer_fill(self):
		if self.capture.isOpened():
			print('Filling buffer...')
			while (self.status) and (self.np_buffer.shape[0] < self.total):
				(self.status, self.frame) = self.capture.read()
				self.np_buffer.append(self.frame)
				self.t.append((self.k, time.time()))
				self.j += 1
				self.k += 1

	def check_ping(self):
		hostname = ip + args.camera
		response = system("ping -c 1 " + hostname)
		# and then check the response...
		if response == 0:
			pingstatus = True
		else:
			pingstatus = False
		return pingstatus


	def con_restore(self):
		self.capture.release()
		while not self.check_ping():
			time.sleep(10)
		self.capture = cv2.VideoCapture(source)


	def	saveArray(self, ar):
		# saves array ar, t = tupple(k, time)
		a = 0
		while a < ar.shape[0]:
			#ar[a] = detector.DisplayDetections(ar[a], self.det_boxes)
			#cv2.rectangle(ar[a], (0, (self.frame_height-10)), (300, self.frame_height), (0,0,0), -1)
			#cv2.putText(ar[a], datetime.utcfromtimestamp(t[1]+a*0.04).strftime('%d/%m/%Y %H:%M:%S.%f')[:-4], (0,self.frame_height-3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1, cv2.LINE_AA)
			self.out.write(ar[a])
			a += 1

	def	DetectFromVideo_rb(self, detector, save_output=False, output_dir='output/'):

		self.mp = int(args.fps)			# maxpixel size
		self.mp1 = round(self.total/2 - self.mp/2) # maxpixel in the middle of the buffer
		self.mp2 = self.mp1 + self.mp
		#self.j =	0
		time1	=	0
		time2	=	1
		t0 = t1 = t2 = t3 = t4 = 0
		self.last_frame = 0
		self.last_frame_recorded = 0
		self.recording = False
		mean = 0
		# limit for detection trigger
		perc30 = 0
		mean_limit = 130
		# number of sec to be added for capture
		sec_post = 0
		self.station = args.station

		mask = False
        # apply the mask if there is any
		maskFile = 'mask-' + args.station + '.bmp'
		if path.exists(maskFile):
			print ('Loading mask...')
			maskImage = Image.open(maskFile).convert('L')
			maskImage = np.array(maskImage, dtype='uint8')
			random_mask = np.random.rand(maskImage.shape[0],maskImage.shape[1],1) * 255
			mask = True
		else:
			print ('No mask file found')
		time.sleep(5)
		while True:
			# if buffer is ready
			if (self.j >= self.mp) and (self.np_buffer.shape[0] >= self.total):
				# new maxpixel frame to be tested for detection
				self.j = 0
				if not self.recording:
					print ("detecting at fps={:2.1f}".format(self.mp/(time.time()-time1)) + ' | ' + str(self.frame_width) + 'x' + str(self.frame_height) + ' | maxpixel=' + str(self.mp) + ' | threshold=' + str(round(detector.Threshold * 10)/10) + ' | t1=' + "{:1.3f}".format(t1-t0) + ' | t2=' + "{:1.3f}".format(t2-t1) + ' | t3=' + "{:1.3f}".format(t3-t2) + ' | t4=' + "{:1.3f}".format(t4-t3) + ' | perc30=' + "{:.0f}".format(perc30) + '  ', end='\r', flush=True)
				time1 =	t0 = time.time()
								
				# timestamp of the 1st frame in the buffer for file name
				t_frame1 = self.t[(self.k - self.total)][1]
				
				# slice the buffer for detection
				buffer_small = self.np_buffer[self.mp1:self.mp2,:,:,:]
				t1 = time.time()
				
				# create maxpixel image
				img = np.max(buffer_small, axis=0)				
				# 1 second slice and maxpixel in a single step by cython, slower than expected
				#img = cumpy.maxpixel(np.asarray(self.np_buffer), self.mp1, self.mp2)
				
				# calculate mean and percentile
				mean = np.copy(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY))
				perc30 = np.percentile(mean, 30)
				
				# apply mask, random mask pattern works best
				if mask:
					#img[maskImage < 3] = np.mean(img[:250,:,:])
					img[maskImage < 3] = random_mask[maskImage < 3]
					#img[maskImage < 3] = np.mean(img[maskImage > 0]) - 20
				t2 = time.time()
				
				# run the detection
				self.det_boxes = detector.DetectFromImage(img)
				t3 = time.time()
				
				# display detection boxes on live view
				img	= detector.DisplayDetections(img, self.det_boxes)
				#cv2.rectangle(img, (0, (self.frame_height-10)), (114, self.frame_height), (0,0,0), -1)
				#cv2.putText(img, datetime.utcfromtimestamp(self.time0 + self.k * 0.04).strftime('%d/%m/%Y %H:%M:%S'), (0,self.frame_height-3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1, cv2.LINE_AA)

				# process if any meteor detected
				if ((self.det_boxes) and (perc30 < mean_limit)):
					if save_output:
						subfolder = 'output/' + args.station + '_' + time.strftime("%Y%m%d", time.gmtime())
						if not path.exists(subfolder):
							mkdir(subfolder)
						output_path = path.join(subfolder, args.station +  '_' + datetime.utcfromtimestamp(t_frame1).strftime("%Y%m%d_%H%M%S_%f") + '.mp4')
						self.out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), 25, (img.shape[1], img.shape[0])) 
						self.last_frame = len(self.t)-3
						buffer = np.copy(self.np_buffer)
						print ("\n" + 'Starting recording...', time.strftime("%H:%M:%S", time.gmtime()), output_path, self.det_boxes, 'frame: ' + str(self.last_frame - buffer.shape[0]) + '-' + str(self.last_frame))
						self.saveArray(buffer)
						self.out.release()
						self.last_frame_recorded = self.last_frame

				# update the screen
				img = cv2.resize(img, (928, 522), interpolation = cv2.INTER_AREA)
				cv2.imshow('TF2 Detection',	img)
				key = cv2.waitKeyEx(1)
				if key == 27:
					#self.out.release()
					break
				elif key == 113:
					detector.Threshold += 0.1
				elif key == 97:
					detector.Threshold -= 0.1
				time2 = t4 = time.time()

		self.capture.release()

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Object Detection from Images or Video')
	parser.add_argument('--model_path', help='Path to frozen detection model', default='trt_fp16_dir/saved_model/')
	parser.add_argument('--path_to_labelmap', help='Path to labelmap (.pbtxt) file', default='labelmap.pbtxt')
	parser.add_argument('--class_ids', help='id of classes to detect, expects string with ids delimited by ","', type=str, default=None) # example input "1,3" to detect person and car
	parser.add_argument('--threshold', help='Detection Threshold', type=float, default=0.7)
	parser.add_argument('--images_dir', help='Directory to input images)', default='data/samples/images/')
	parser.add_argument('--video_path', help='Path to input video)', default='data/samples/pedestrian_test.mp4')
	parser.add_argument('--output_directory', help='Path to output images and video', default='output/')
	parser.add_argument('--video_input', help='Flag for video input', default=False, action='store_true')  # default is false
	parser.add_argument('--save_output', help='Flag for save images and video with detections visualized', default=True, action='store_true')  # default is false
	parser.add_argument('--camera', help='camera number', default='10')
	parser.add_argument('--station', help='station name', default='XX0XXXX')
	parser.add_argument('--fps', help='fps', default=25)
	args = parser.parse_args()

	# IP segment of cameras
	ip = '192.168.150.'
	# camera connection string
	source = 'rtsp://' + ip + args.camera + ':554/user=admin&password=uiouio&channel=1&stream=0'
	station = args.station

	id_list = None
	if args.class_ids is not None:
		id_list = [int(item) for item in args.class_ids.split(',')]

	if args.save_output:
		if not path.exists(args.output_directory):
			makedirs(args.output_directory)

	# instance of the class DetectorTF2
	print("Starting detector...")
	detector = DetectorTF2(args.model_path, args.path_to_labelmap, class_id=id_list, threshold=args.threshold)

	if args.video_input:
		DetectFromVideo(detector, args.video_path, save_output=args.save_output, output_dir=args.output_directory)
	else:
		# start the capture and wait to fill the buffer
		video_stream_widget = VideoStreamWidget(source)
		time.sleep(8)
		# start detector
		video_stream_widget.DetectFromVideo_rb(detector, save_output=args.save_output, output_dir=args.output_directory)
		
	print("Done ...")
	cv2.destroyAllWindows()
