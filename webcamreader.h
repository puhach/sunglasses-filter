#ifndef WEBCAMREADER_H
#define WEBCAMREADER_H



#include "mediasource.h"


#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

#include <string>



// A class for reading frames from a webcam; implements a universal media source API

class WebcamReader: public MediaSource
{
public:
	WebcamReader(int cameraId = 0)
		: MediaSource(MediaSourceType::Webcam, std::string("cam:").append(std::to_string(cameraId)), false)
		, cap(cameraId)
		, cameraId(cameraId)
	{
		CV_Assert(this->cap.isOpened());
	}

	cv::Size getFrameSize() const;

	virtual bool readNext(cv::Mat& frame) override;

	virtual void reset() override;

private:
	cv::VideoCapture cap;
	int cameraId;
};	// WebcamReader


#endif	// WEBCAMREADER_H