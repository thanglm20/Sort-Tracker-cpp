
# 1.Introduction
App uses ncnn to detector traffic then tracks object with Kalman filter and Hungarian Algorithm
# 2.Requirements
	std c++14
	OpenCV 4.5.0 built from source with c++.
	sudo apt-get install build-essential cmake pkg-config
	sudo apt-get install libjpeg-dev libtiff5-dev libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev x264 v4l-utils libhdf5-dev libprotobuf-dev protobuf-compiler git libgtk2.0-dev libgtk-3-dev
# 3.Build & Run
Some commands are:
```
mkdir build
cd build
cmake ..
make
./tracker
```
# 4.References
https://github.com/Tencent/ncnn


