#ifndef VIDEOFILEREADER_H
#define VIDEOFILEREADER_H


#include "mediasource.h"


#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

#include <string>


// A class for reading video files which implements a universal media source API

class VideoFileReader: public MediaSource
{
public:
	VideoFileReader(const std::string& videoFile, bool looped = false);
	
	cv::Size getFrameSize() const;

	virtual bool readNext(cv::Mat& frame) override;

	virtual void reset() override;

private:
	cv::String inputFile;
	cv::VideoCapture cap;
};	// VideoFileReader



#endif	// VIDEOFILEREADER_H