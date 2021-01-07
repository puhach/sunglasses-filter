#ifndef DUMMYWRITER_H
#define DUMMYWRITER_H


#include "mediasink.h"


// A dummy writer provides the implementation for a universal media sink API that does nothing

class DummyWriter: public MediaSink
{
public:
	DummyWriter() : MediaSink(MediaSinkType::Dummy, "") {}
	
	virtual void write(const cv::Mat& frame) override { }
};	// DummyWriter



#endif	// DUMMYWRITER_H