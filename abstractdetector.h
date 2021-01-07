#ifndef ABSTRACTDETECTOR_H
#define ABSTRACTDETECTOR_H



#include <opencv2/core.hpp>

#include <vector>
#include <memory>



// An abstract parent class for single-class bounding box detectors

class AbstractDetector
{
public:
	virtual ~AbstractDetector() = default;

	/*virtual*/ void detect(const cv::Mat& image, std::vector<cv::Mat>& objs);	// provided for convenience

	virtual void detect(const cv::Mat &image, std::vector<cv::Rect>& rects) = 0;

	// C.130: For making deep copies of polymorphic classes prefer a virtual clone function instead of copy construction/assignment:
	// http://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#Rh-copy
	virtual std::unique_ptr<AbstractDetector> clone() const & = 0;
	virtual std::unique_ptr<AbstractDetector> clone() && = 0;	// probably, not needed since we are using pointers, which can be moved anyway

protected:

	// C.67: A polymorphic class should suppress copying:
	// http://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#Rc-copy-virtual
	AbstractDetector() = default;
	AbstractDetector(const AbstractDetector&) = default;
	AbstractDetector(AbstractDetector&&) = default;
	
	// Explicitly mark the assignment operators as deleted
	AbstractDetector& operator = (const AbstractDetector&) = delete;
	AbstractDetector& operator = (AbstractDetector&&) = delete;
};	// AbstractDetector



#endif	// ABSTRACTDETECTOR_H