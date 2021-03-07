# MeteorDL - Meteor Deep Learning detector

MeteorDL has been written as a primary object detection tool for RMS:
https://github.com/CroatianMeteorNetwork/RMS.
However, it can be used as standalone tool.

Requirements:
- linux OS
- NVidia CUDA capable GPU
- min. 4GB RAM
- python > 3.5

It has been tested on linux PC with NVidia GTX-1080 GPU and Jetson Nano 4GB.

**Installation**
- Install the Tensorflow 2 Object detection API v2:
  https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2.md

- Install following additional python modules:
  - numpy
  - matplotlib
  - dvg_ringbuffer
  - Pillow (PIL)

- Once everything is ready, clone the repository to any folder and run for example
  python meteordl.py --camera 10 --station CZ0001 --fps 25
  
  where:
    - camera = last digit of the IP address. The IP segment of the address can be edited directly in the script
    - station = RMS station designation. If none given, the default is used
    - fps = framer per second camera settings, currently 10-25 supported

**Live view**
detection live view is provided including some basic run time parameters on the command line. 

**Detection model**
detection nmodel is based on SSD MobileNet V2 FPNLite 640x640 pretrained model, retrained on custom meteor data.
The procedure for data preparation and training is described on:


