#include "videofilewriter.h"




void VideoFileWriter::write(const cv::Mat& frame)
{
	writer.write(frame);
}	// write

