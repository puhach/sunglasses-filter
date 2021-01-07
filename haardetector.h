#ifndef HAARDETECTOR_H
#define HAARDETECTOR_H


#include "abstractdetector.h"

#include <opencv2/core.hpp>
#include <opencv2/objdetect.hpp>

#include <string>
#include <vector>
#include <memory>



// A Haar-based single-class bounding box detector  

class HaarDetector: public AbstractDetector
{
public:

	HaarDetector(const cv::String &fileName, double scaleFactor = 1.1, int minNeighbors = 3, 
		int flags = 0, cv::Size minSize = cv::Size(), cv::Size maxSize = cv::Size());

	HaarDetector() = default;

	double getScaleFactor() const noexcept { return this->scaleFactor; }
	void setScaleFactor(double scaleFactor) noexcept { this->scaleFactor = scaleFactor; }

	int getMinNeighbors() const noexcept { return this->minNeighbors; }
	void setMinNeighbors(int minNeighbors) noexcept { this->minNeighbors = minNeighbors; }

	int getFlags() const noexcept { return this->flags; }
	void setFlags(int flags) noexcept { this->flags = flags; }

	cv::Size getMinSize() const { return this->minSize; }
	void setMinSize(const cv::Size& minSize) { this->minSize = minSize; }

	cv::Size getMaxSize() const { return this->maxSize; }
	void setMaxSize(const cv::Size& maxSize) { this->maxSize = maxSize; }

	virtual void detect(const cv::Mat &image, std::vector<cv::Rect>& rects) override;

	virtual std::unique_ptr<AbstractDetector> clone() const & override;
	virtual std::unique_ptr<AbstractDetector> clone() && override;

protected:
	HaarDetector(const HaarDetector&) = default;
	HaarDetector(HaarDetector&&) = default;

private:
	cv::CascadeClassifier cascadeClassifier;
	double scaleFactor;
	int minNeighbors;
	int flags;
	cv::Size minSize, maxSize;
};	// HaarFaceDetector


#endif	// HAARDETECTOR_H