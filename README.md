# MeteorDL - Meteor Deep Learning detector

MeteorDL has been written as a primary real-time object detection tool for RMS, software used in Global Meteor Network:

https://github.com/CroatianMeteorNetwork/RMS

https://globalmeteornetwork.org/

However, it can be used as standalone tool.
It captures the IP camera stream and detects meteors (or other phenomena seen on dark sky) in real time, featuring 5 second ring-buffer, including 2 second pre-event data.
The detection is performed on 1 second maxpixel image in the middle of the ring buffer.

**Requirements:**

- Linux OS
- NVidia CUDA capable GPU
- min. 4GB RAM
- python > 3.5

It has been tested on Linux PC with NVidia GeForce GTX-1080 GPU and Jetson Nano 4GB.

**Installation**
- Install the Tensorflow 2 Object detection API v2:
  https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2.md

- Install following additional python modules:
  - numpy
  - matplotlib
  - dvg_ringbuffer
  - Pillow (PIL)

- Once everything is ready, clone the repository to any folder and run for example:
  
  python meteorDL-nano.py --camera 10 --station CZ0001 --fps 25
  
  where:
    - camera = last digit of the IP address. The IP segment of the address can be edited directly in the script
    - station = RMS station designation. If none given, the default XX0XXXX is used
    - fps = framer per second camera settings, currently 10-25 supported
  
  The startup procedure takes several minutes (on Jetson Nano) until the live screen is shown and detection started

**Live view**

detection live view is provided, including basic run time parameters on the command line. 

**Detection model**

detection nmodel is based on SSD MobileNet V2 FPNLite 640x640 pretrained model, retrained on custom meteor data in VOC format.
The procedure for data preparation and training is described on:

https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md

and

https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_training_and_evaluation.md

**Mask file**

to avoid false detections on camera installation with lower elevation in urban areas, the mask file has to be provided.
The expected file name is:
> mask-<station-name>.bmp
  
**Multi-camera detection**

MeteorDL has been tested on linux PC (16-core 2.5GHz CPU, 32 GB RAM) with two full-HD cameras simultaneously without any issues.
On Jetson Nano, the resolution tested was 1280x720.

