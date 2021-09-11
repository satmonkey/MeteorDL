![output](https://user-images.githubusercontent.com/5328519/110521871-97224f00-8110-11eb-94be-b13ebe622a85.jpg)
# MeteorDL - Meteor Deep Learning detector

MeteorDL has been written primarily as a real-time object detection tool for RMS, software used in Global Meteor Network:

https://github.com/CroatianMeteorNetwork/RMS

https://globalmeteornetwork.org/

However, it can be used as standalone tool.

MeteorDL captures the IP camera stream and detects meteors (or other phenomena seen in the dark sky) in real time, featuring ring-buffer, and preconfigured pre- and post-event data period.
Any camera suitable for Opencv (FFMpeg or Gstreamer) is fine.
The detection is performed on 1 second maxpixel image.
In case of positive detection, corresponding several second video chunk is created and saved in the output directory as MP4 video.
The PC version is using GPU-based ring buffer, while Jetson Nano and Pi4 version is using dvg-ringbuffer (CPU based, cythonized).

Raspberry PI4 version needs Google Coral TPU USB stick:

https://coral.ai/products/accelerator/

## Requirements:

- SW: Linux OS
- HW: 
  - Option 1: PC + NVidia CUDA capable GPU
  - Option 2: Jetson nano
  - Option 3: RPi4 + Coral Edge TPU USB accelerator
- min. 4GB RAM
- python > 3.5
- IP, USB or MIPI camera with 720p or 1080p resolution

It has been tested on Linux PC with NVidia GeForce GTX-1080 GPU, Jetson Nano 4GB, and RPi4 4GB + Coral TPU

## Installation
- Install the Tensorflow 2 Object detection API v2 (not needed for RPi4 version)
  
  https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2.md
  
  or
  
  https://gilberttanner.com/blog/tensorflow-object-detection-with-tensorflow-2-creating-a-custom-model
  
- Install tflite_runtime (RPi4 version)

  `echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list`
  
  `curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -`
  
  `sudo apt-get update`
  
  `pip3 install --index-url https://google-coral.github.io/py-repo/ tflite_runtime`

- Install Edge TPU runtime and Pycoral library (RPi4 version):

  `sudo apt-get install libedgetpu1-std`
  
  `sudo apt-get install python3-pycoral`

- Install following additional python modules via e.g. pip install:
  - numpy
  - matplotlib
  - dvg_ringbuffer (Jetson Nano and RPi4)
  - cupy (x86 version)
  - Pillow (PIL)
  - Opencv v4, with gstreamer and FFmpeg backend

- Once everything is ready, clone the repository to any folder, edit config.ini to suit your configuration and run for example:
  
  `python meteorDL-nano.py --station CZ0001`
  or
  `python meteorDL-x86.py --station CZ0001`
  or
  `python meteorDL-pi.py --station CZ0001`
  
  where station = RMS station designation. If none given, the default XX0XXXX is used.
  
  The startup procedure takes several minutes (on Jetson Nano) until the live screen is shown and detection started

## Live view

detection live view is provided, including basic run time parameters on the command line. 

## Detection model

model used is the object detection model based on SSD MobileNet V2 FPNLite 640x640 pretrained model, retrained on low amount (~1000 of meteor images) custom meteor data in VOC format, along with deep augmentation preprocessing.
The general procedure for data preparation and training is described on:

https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md

and

https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_training_and_evaluation.md

## Mask file

to avoid false detections on camera installation with lower elevation in urban areas, the mask file has to be provided.
The expected file name is: `mask-<station-name>.bmp`
  
## Multi-camera detection

MeteorDL has been tested on linux PC (16-core 2.5GHz CPU, 32 GB RAM) with two full-HD cameras simultaneously without any issues.
On Jetson Nano, the resolution tested was 1280x720.

## Run time interaction

following keys are active for interaction during the run time:

- ESC - exit

## False detections

Dynamic sensitivity threshold filters out possible false detections (fast moving clouds, nearby planes, etc). Therefore, only sudden and quick events should get detected, like meteors and TLE. Still, you may observe some false positive detections, especially in complex sky conditions around full moon phase.  



