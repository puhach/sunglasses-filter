#include "imagefilewriter.h"


void ImageFileWriter::write(const cv::Mat& frame)
{
	CV_Assert(frame.size() == this->frameSize);
	if (!cv::imwrite(getMediaPath(), frame))
		throw std::runtime_error("Failed to write the output image.");
}	// write

