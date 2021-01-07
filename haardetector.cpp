#include "haardetector.h"

#include <opencv2/imgproc.hpp>


HaarDetector::HaarDetector(const cv::String &fileName, double scaleFactor, int minNeighbors, int flags, cv::Size minSize, cv::Size maxSize)
	: cascadeClassifier(fileName)
	, scaleFactor(scaleFactor)
	, minNeighbors(minNeighbors)
	, flags(flags)
	, minSize(std::move(minSize))
	, maxSize(std::move(maxSize))
{
	CV_Assert(!this->cascadeClassifier.empty());
}

void HaarDetector::detect(const cv::Mat &image, std::vector<cv::Rect>& rects)
{
	CV_Assert(!image.empty());

	// Histogram equalization considerably improves eye detection
	cv::Mat imageGray;
	cv::cvtColor(image, imageGray, cv::COLOR_BGR2GRAY);
	cv::equalizeHist(imageGray, imageGray);

	this->cascadeClassifier.detectMultiScale(imageGray, rects, this->scaleFactor, this->minNeighbors, this->flags, this->minSize, this->minSize);
}	// detect

std::unique_ptr<AbstractDetector> HaarDetector::clone() const &
{
	// Although make_unique can't access our protected copy constructor, it must be safe to use new here, 
	// because only one object is created this way and nothing will leak even if it throws an exception
	return std::unique_ptr<HaarDetector>(new HaarDetector(*this));
	//return std::make_unique<HaarDetector>(*this);
}

std::unique_ptr<AbstractDetector> HaarDetector::clone()&&
{
	return std::unique_ptr<HaarDetector>(new HaarDetector(std::move(*this)));
}
