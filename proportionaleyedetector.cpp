#include "proportionaleyedetector.h"




void ProportionalEyeDetector::detect(const cv::Mat& face, std::vector<cv::Rect>& rects)
{
	// Find the eye regions using typical face proportions
	int x1 = face.cols / 5, x2 = 3*face.cols/5;
	int y = face.rows / 3;
	int h = face.rows / 8;
	int w = face.cols / 5;

	rects.reserve(2);
	rects.emplace_back(x1, y, w, h);
	rects.emplace_back(x2, y, w, h);
}	// detect

std::unique_ptr<AbstractDetector> ProportionalEyeDetector::clone() const &
{
	// Since we declared the copy constructor as protected, make_unique doesn't work right out of the box.
	// Making it a friend may be problematic, but using new should be safe here, because we are constructing only one object.
	// https://stackoverflow.com/questions/33905030/how-to-make-stdmake-unique-a-friend-of-my-class
	// https://stackoverflow.com/questions/33905782/error-on-msvc-when-trying-to-declare-stdmake-unique-as-friend-of-my-templated
	return std::unique_ptr<ProportionalEyeDetector>(new ProportionalEyeDetector(*this));
	// return std::make_unique<ProportionalEyeDetector>(*this);
}

std::unique_ptr<AbstractDetector> ProportionalEyeDetector::clone() &&
{
	return std::unique_ptr<ProportionalEyeDetector>(new ProportionalEyeDetector(std::move(*this)));
}
