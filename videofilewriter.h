#ifndef VIDEOFILEWRITER_H
#define VIDEOFILEWRITER_H



#include "mediasink.h"

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

#include <string>



// A class for writing video files which implements a universal media sink API

class VideoFileWriter: public MediaSink
{
public:
	VideoFileWriter(const std::string& videoFile, cv::Size frameSize, const char(&fourcc)[4], double fps)
		: MediaSink(MediaSinkType::VideoFile, videoFile)
		, writer(videoFile, cv::VideoWriter::fourcc(fourcc[0], fourcc[1], fourcc[2], fourcc[3]), fps, std::move(frameSize), true) {	}

	virtual void write(const cv::Mat& frame) override;

private:
	cv::VideoWriter writer;
};	// VideoWriter



#endif	// VIDEOFILEWRITER_H