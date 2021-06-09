# by Milan Kalina
# based on Tensorflow API v2 Object Detection code
# https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2.md

from os import path, system, mkdir, makedirs
import cv2
import time
import os
import argparse
import numpy as	np
from datetime import datetime
from threading import Thread, Semaphore, Lock
from detector_tflite import DetectorTF2
import dvg_ringbuffer as rb
import configparser
from PIL import Image
import statistics as st
import multiprocessing as mp
import subprocess as sp
from queue import Queue
import cumpy as cp

#import numpy as np
#import time



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
			# Read the next frame from the stream in a different thread, builds and rotates the buffer
			# and maintains buffer - list of consequtive frames
			self.total = int(args.fps) * 5	# buffer size
			self.k = 0			# global frame counter
			self.j = 0			# maxpixel counter
			self.t = [(0,0)]	# time tracking
			#mutex = Lock()
			self.time0 = time.time()
			(self.status, self.frame) = self.capture.read()
			self.frame_width = self.frame.shape[1]
			self.frame_height = self.frame.shape[0]
			self.np_buffer = rb.RingBuffer(self.total, dtype=(np.uint8,(self.frame_height, self.frame_width, 3)), allow_overwrite=False)
			print("Filling the ring buffer...")
			self.buffer_fill()
			print("Buffer filled, starting..." + str(self.np_buffer.shape))
			
			while True:
				if self.capture.isOpened():
					(self.status, self.frame) = self.capture.read()
					#self.frame = self.frame[self.border:self.border+720,:]
					if (self.status):
						cv2.rectangle(self.frame, (0, (self.frame_height-10)), (200, self.frame_height), (0,0,0), -1)
						cv2.putText(self.frame, station + ' ' + datetime.utcfromtimestamp(time.time()).strftime('%d/%m/%Y %H:%M:%S.%f')[:-4] + ' ' + str(self.k), (1,self.frame_height-3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1, cv2.LINE_AA)
						#mutex.acquire()
						self.np_buffer.popleft()
						self.np_buffer.append(self.frame)
						#mutex.release()
						self.t = self.t[1:]
						#self.t = np.roll(self.t, -1, axis=0)
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
			while (self.status) and (len(self.t) < self.total+1):
				(self.status, self.frame) = self.capture.read()
				print(str(self.frame.shape) + '  ' + str(self.j) + '  ', end='\r', flush=True)
				self.np_buffer.append(self.frame)
				self.t.append((self.k, time.time()))
				self.j += 1
				self.k += 1

	def check_ping(self):
		#hostname = ip
		response = system("ping -c 1 " + ip)
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


	def saveArray(self):
		while True:
			# saves numpy array into video file
			a = 0
			# Lock the resources until free access to movie file
			self.threadlock.acquire()
			# wait until buffer received from queue
			ar = self.q.get()
			while a < ar.shape[0]:
				#ar[a] = detector.DisplayDetections(ar[a], self.det_boxes)
				#cv2.rectangle(ar[a], (0, (self.frame_height-10)), (300, self.frame_height), (0,0,0), -1)
				#cv2.putText(ar[a], datetime.utcfromtimestamp(t[1]+a*0.04).strftime('%d/%m/%Y %H:%M:%S.%f')[:-4], (0,self.frame_height-3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1, cv2.LINE_AA)
				self.out.write(ar[a])
				a += 1
			self.threadlock.release()
			self.q.task_done()
			
		
	def DetectFromStream(self, detector, save_output=False, output_dir='output/'):

		self.mp = int(args.fps)			# maxpixel size
		self.mp1 = round(self.total/2 - self.mp/2) # maxpixel in the middle of the buffer
		self.mp2 = self.mp1 + self.mp
		#self.j = 0
		time1 = 0
		time2 = 1
		t0 = t1 = t2 = t3 = t4 = 0
		self.last_frame = 0
		self.last_frame_recorded = 0
		self.recording = False
		mean = 0
		# limit for detection trigger
		perc30 = 0
		mean_limit = 200
		bg_max = 0.9
		thr = 0.9
		bg=[1,1,1,1,1,1,1,1,1]
		# number of sec to be added for capture
		#sec_post = 0
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
		# start a queue for saving threads, then start a thread
		self.q = Queue()
		self.threadlock=Lock()
		Thread(target=self.saveArray, daemon=True).start()
		
		
		
		time.sleep(5)
		while True:
			# if new 1s chunk in the buffer is ready for detection
			if (self.j >= self.mp) and (self.np_buffer.shape[0] >= self.total):
				# new maxpixel frame to be tested for detection
				self.j = 0
				print ("detecting at fps={:2.1f}".format(self.mp/(time.time()-time1)) + ' | t=' + str(int(self.t[-1][0])) + ' | buffer=' + str(self.np_buffer.shape) + ' | maxpixel=' + str(self.mp) + ' | threshold=' + str(round((bg_max + margin) * 100)/100) + ' | t1=' + "{:1.3f}".format(t1-t0) + ' | t2=' + "{:1.3f}".format(t2-t1) + ' | t3=' + "{:1.3f}".format(t3-t2)+ ' | t4=' + "{:1.3f}".format(t4-t3) + ' | perc30=' + "{:.0f}".format(perc30) + '  ', end='\r', flush=True)
				time1 = t0 = time.time()
				# timestamp for file name, 1st frame of maxpixel image
				t_frame1 = self.t[0][1]
					
				# take 1s from middle of buffer to create maxpixel for detection, green channel omly for better performance
				buffer_small = self.np_buffer[self.mp1:self.mp2,:,:,[1]]
				
				t1 = time.time()
				# calculate the maxpixel image
				img = np.max(buffer_small, axis=0)
				t2 = time.time()
				perc30 = np.percentile(img, 30)
				img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

				if mask:
					# apply trick from RMS
					img[maskImage < 3] = random_mask[maskImage < 3]
				t3 = time.time()
				self.det_boxes = detector.DetectFromImage(img)
				img_clean = img
				if self.det_boxes[0][5] > 0.1:
					img = detector.DisplayDetections(img, self.det_boxes[:5])

				t4 = time.time()
				img_small = cv2.resize(img, (928, 522), interpolation = cv2.INTER_AREA)
				cv2.imshow('Meteor detection', img_small)
				key = cv2.waitKeyEx(1)
				
				# trigger the saving if signal above the mean noise and sky background below the daytime brightness 
				if (self.det_boxes and perc30 < mean_limit):
					if self.det_boxes[0][5] > (bg_max + margin):
						if save_output:
							# prepare file and folder for saving
							subfolder = 'output/' + args.station + '_' + time.strftime("%Y%m%d", time.gmtime())
							if not os.path.exists(subfolder):
								os.mkdir(subfolder)
							self.output_path = os.path.join(subfolder, station +  '_' + datetime.utcfromtimestamp(t_frame1).strftime("%Y%m%d_%H%M%S_%f") + '_' + "{:0.0f}".format(100*self.det_boxes[0][5]) + '.mp4')
							self.out = cv2.VideoWriter(self.output_path, cv2.VideoWriter_fourcc(*"mp4v"), 25, (img.shape[1], img.shape[0]))
							
							# note the last frame
							self.last_frame = self.t[-1][0]
							# get the buffer copy to be saved
							buffer = np.copy(self.np_buffer)
							# create another saving task
							self.q.put(buffer)
							
							print ("\n" + 'Starting recording...', datetime.utcfromtimestamp(time.time()).strftime("%H:%M:%S.%f"), self.output_path, self.det_boxes[0], 'frame: ' + str(self.last_frame - buffer.shape[0]) + '-' + str(self.last_frame))
							
							output_path_mp = os.path.join(subfolder, station +  '_mp_' + datetime.utcfromtimestamp(t_frame1).strftime("%Y%m%d_%H%M%S_%f") + '.jpg')
							output_path_mp_clean = os.path.join(subfolder, station +  '_mp-clean_' + datetime.utcfromtimestamp(t_frame1).strftime("%Y%m%d_%H%M%S_%f") + '.jpg')
							# save the maxpixel as jpg
							cv2.imwrite(output_path_mp, img)
							cv2.imwrite(output_path_mp_clean, img_clean)
							self.last_frame_recorded = self.last_frame

							# save another <post> seconds of frames
							for s in range(3):
								# wait until new data available
								while self.t[-1][0] < (self.last_frame + 3*self.mp):
									...
								if self.t[-1][0] == (self.last_frame + 3*self.mp):
										self.last_frame = self.t[-1][0]
										buffer = np.copy(self.np_buffer[-3*self.mp:])
										self.q.put(buffer)
										print(f"Saving additional chunk: {self.last_frame-3*self.mp}-{self.last_frame}")
										self.last_frame_recorded = self.last_frame
							self.q.join()
							self.out.release()
							#print(f"Writer released...")
				
				# update mean and max noise
				bg = bg[-9:]
				bg.append(self.det_boxes[0][5])
				bg_max = max(bg[:9])
				
				# update the screen
				#img = cv2.resize(img, (928, 522), interpolation = cv2.INTER_AREA)
				#cv2.imshow('Meteor detection', img)
			
				# key handling, key codes valid pro RPi4
				key = cv2.waitKeyEx(1)
				if key == 1048603:
					break
				elif key == 1048689:
					detector.Threshold += 0.1
				elif key == 1048673:
					detector.Threshold -= 0.1
				time2 = time.time()

		self.capture.release()

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Object Detection from Images or Video')
	parser.add_argument('--model_path', help='Path to frozen detection model', default='tflite-real-640-4/model_full_integer_quant_edgetpu.tflite')
	parser.add_argument('--path_to_labelmap', help='Path to labelmap (.txt) file', default='labelmap.txt')
	parser.add_argument('--class_ids', help='id of classes to detect, expects string with ids delimited by ","', type=str, default=None) # example input "1,3" to detect person and car
	parser.add_argument('--threshold', help='Detection Threshold', type=float, default=0.01)
	parser.add_argument('--images_dir', help='Directory to input images)', default='data/samples/images/')
	parser.add_argument('--video_path', help='Path to input video)', default='data/samples/pedestrian_test.mp4')
	parser.add_argument('--output_directory', help='Path to output images and video', default='output/')
	parser.add_argument('--video_input', help='Flag for video input', default=False, action='store_true')  # default is false
	parser.add_argument('--save_output', help='Flag for save images and video with detections visualized', default=True, action='store_true')  # default is false
	parser.add_argument('--camera', help='camera number', default='10')
	parser.add_argument('--station', help='station name', default='XX0XXXX')
	parser.add_argument('--fps', help='fps', default=25)
	args = parser.parse_args()

	margin = 0.3
	
	station = args.station
	config = configparser.ConfigParser()
	config.read('config.ini')
	if station not in config:
		station = 'default'
	fps = int(config[station]['fps'])
	ip = config[station]['ip']
	
	# on Rpi4, FFMPEG backend is preffered
	source = 'rtsp://' + config[station]['ip'] + ':' + config[station]['rtsp']
	
	print(f"Streaming from {source}")
	sec_post = int(config['general']['post_seconds'])
	sec_pre = int(config['general']['pre_seconds'])
	b_size = int(config['general']['buffer_size'])
		
	id_list = None
	if args.class_ids is not None:
		id_list = [int(item) for item in args.class_ids.split(',')]

	if args.save_output:
		if not path.exists(args.output_directory):
			makedirs(args.output_directory)

	# instance of the class DetectorTF2
	print("Starting detector...")
	detector = DetectorTF2(args.model_path, args.path_to_labelmap, class_id=id_list, threshold=args.threshold)
	print("detector started...")

	if args.video_input:
		DetectFromVideo(detector, args.video_path, save_output=args.save_output, output_dir=args.output_directory)
	else:
		# start the capture and wait to fill the buffer
		video_stream_widget = VideoStreamWidget(source)
		time.sleep(8)
		# start detector
		video_stream_widget.DetectFromStream(detector, save_output=args.save_output, output_dir=args.output_directory)
		
	print("Done ...")
	cv2.destroyAllWindows()
