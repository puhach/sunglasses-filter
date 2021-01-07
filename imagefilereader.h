#ifndef IMAGEFILEREADER_H
#define IMAGEFILEREADER_H


#include "mediasource.h"

#include <opencv2/core.hpp>

#include <string>



// A class for reading image files which implements a universal media source API

class ImageFileReader: public MediaSource
{
public:
	ImageFileReader(const std::string& imageFile, bool looped = false);
	
	virtual cv::Size getFrameSize() const;

	bool readNext(cv::Mat& frame) override;
	
	void reset() override;

private:
	mutable cv::Mat cache;
};	// ImageFileReader


#endif	// IMAGEFILEREADER_H