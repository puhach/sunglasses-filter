#include "abstractdetector.h"




void AbstractDetector::detect(const cv::Mat& image, std::vector<cv::Mat>& objs)
{
	std::vector<cv::Rect> rects;
	detect(image, rects);	// virtual

	objs.reserve(rects.size());
	for (const cv::Rect& r : rects)
	{
		objs.push_back(image(r));
	}
}	// detect
