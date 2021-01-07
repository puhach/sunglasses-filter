#ifndef IMAGEFILEWRITER_H
#define IMAGEFILEWRITER_H



#include "mediasink.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <string>
#include <stdexcept>



// A class for writing image files which implements a universal media sink API

class ImageFileWriter: public MediaSink
{
public:
	ImageFileWriter(const std::string& imageFile, cv::Size frameSize)
		: MediaSink(MediaSinkType::ImageFile, cv::haveImageWriter(imageFile) ? imageFile : throw std::runtime_error("No encoder for this image file: " + imageFile))
		, frameSize(std::move(frameSize)) { }

	virtual void write(const cv::Mat& frame) override;

private:
	cv::Size frameSize;
};	// ImageFileWriter


#endif	// IMAGEFILEWRITER_H