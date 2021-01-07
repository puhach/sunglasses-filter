#ifndef PROPORTIONALEYEDETECTOR_H
#define PROPORTIONALEYEDETECTOR_H



#include "abstractdetector.h"


#include <opencv2/core.hpp>

#include <vector>
#include <memory>




// Proportional eye detector computes eye locations using typical proportions of a frontal face

class ProportionalEyeDetector: public AbstractDetector
{
	// https://stackoverflow.com/questions/33905782/error-on-msvc-when-trying-to-declare-stdmake-unique-as-friend-of-my-templated
	//friend std::unique_ptr<ProportionalEyeDetector> std::make_unique<ProportionalEyeDetector>(const ProportionalEyeDetector&);
public:
	ProportionalEyeDetector() = default;
	~ProportionalEyeDetector() = default;

	virtual void detect(const cv::Mat &face, std::vector<cv::Rect> &rects) override;

	virtual std::unique_ptr<AbstractDetector> clone() const & override;
	virtual std::unique_ptr<AbstractDetector> clone() && override;

protected:
	// C.67: A polymorphic class should suppress copying:
	// http://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#Rc-copy-virtual
	ProportionalEyeDetector(const ProportionalEyeDetector&) = default;
	ProportionalEyeDetector(ProportionalEyeDetector&&) = default;
};	// ProportionalEyeDetector



#endif	// PROPORTIONALEYEDETECTOR_H