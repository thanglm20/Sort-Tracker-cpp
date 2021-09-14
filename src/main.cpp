

/*
    Module: Object Tracking
    Author: Le Manh Thang
    Created: August 26th, 2021
*/

#include "CrosslineDetector.hpp"
#include "ObjectTracker.hpp"
#include "MotionDetector.hpp"
using namespace std;
using namespace cv;

int main()
{
	// object detector
	std::string namesPath = "../models/traffic.names";
    std::string cfgPath = "../models/traffic-tiny.param";
    std::string weightPath = "../models/traffic-tiny.bin";
	// std::string namesPath = "../models/people.txt";
    // std::string cfgPath = "../models/people.param";
    // std::string weightPath = "../models/people.bin";
	ObjectDetector* detector = new ObjectDetector(namesPath, cfgPath, weightPath);
	CrosslineDetector* crossline = new CrosslineDetector();
	MotionDetector motion;
	ObjectTracker tracker;
	cv::VideoCapture cap;
	while(1)
	{
	cap.open("../videos/traffic.mp4");
	//cap.open("/media/thanglmb/Bkav/AICAM/Testing/videos/tuning.mp4");
	if(!cap.isOpened())
	{
		printf("open video error\n");
		return 0 ;
	}
	cv::Mat frame;
	
	while(1)
	{

		cap >> frame;
		if (frame.empty()) 
            break;
		std::vector<ObjectTrace> detected;
		detector->processDetect(frame, 0.5, detected);
		double start = getTickCount();
		vector<outDataCrossline> output_crossline;
		crossline->setCrossline(cv::Point(200, 400), cv::Point(frame.cols, 400), "all");
    	crossline->updateCrossline(frame, detected, output_crossline);
		// show result
		resize(frame, frame, Size(1280, 720));
        imshow("VideoCapture", frame);
        char key = waitKey(1);
        if(key == 'q') 
        {
            printf("Quited\n");
            break;
        }
        if(key == 'p') 
        {
            printf("Paused\n");
            while(waitKey(1) != 'p');
        }
	}
	}
	return 0;
}
