#include "webcamreader.h"




cv::Size WebcamReader::getFrameSize() const
{
	int w = static_cast<int>(this->cap.get(cv::CAP_PROP_FRAME_WIDTH));
	int h = static_cast<int>(this->cap.get(cv::CAP_PROP_FRAME_HEIGHT));
	return w > 0 && h > 0 ? cv::Size(w, h) : throw std::runtime_error("Failed to get the size of the webcam frame.");
}

bool WebcamReader::readNext(cv::Mat& frame)
{
	return cap.read(frame);
}

void WebcamReader::reset()
{
	this->cap.release();
	CV_Assert(this->cap.open(this->cameraId));
}


