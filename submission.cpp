
#include <iostream>
#include <exception>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/objdetect.hpp>

class ImageFilter
{
public:
	virtual ~ImageFilter() = default;

	virtual void applyInPlace(cv::Mat& image) = 0;

	virtual cv::Mat apply(const cv::Mat& image) = 0;
};	// ImageFilter


class Detector
{
public:
	virtual ~Detector() = default;

	virtual void detect(const cv::Mat &image, std::vector<cv::Mat>& rois) = 0;

	virtual void detect(const cv::Mat &image, std::vector<cv::Rect>& rects) = 0;
};	// Detector

class HaarDetector : public Detector
{
public:

	HaarDetector(const cv::String &fileName, double scaleFactor = 1.1, double minNeighbors = 3, 
		int flags = 0, cv::Size minSize = cv::Size(), cv::Size maxSize = cv::Size());

	// TODO: define getters and setters

	virtual void detect(const cv::Mat &image, std::vector<cv::Mat>& rois) override;

	virtual void detect(const cv::Mat &image, std::vector<cv::Rect>& rects) override;

private:
	cv::CascadeClassifier cascadeClassifier;
	double scaleFactor, minNeighbors;
	int flags;
	cv::Size minSize, maxSize;
};	// HaarFaceDetector

HaarDetector::HaarDetector(const cv::String &fileName, double scaleFactor, double minNeighbors, int flags, cv::Size minSize, cv::Size maxSize)
	: cascadeClassifier(fileName)
	, scaleFactor(scaleFactor)
	, minNeighbors(minNeighbors)
	, flags(flags)
	, minSize(std::move(minSize))
	, maxSize(std::move(maxSize))
{
	CV_Assert(!this->cascadeClassifier.empty());
}

void HaarDetector::detect(const cv::Mat &image, std::vector<cv::Mat>& rois)
{
	// TODO
}

void HaarDetector::detect(const cv::Mat &image, std::vector<cv::Rect>& rects)
{
	// TODO
}


class SunglassesFilter : public ImageFilter
{
public:
	SunglassesFilter(const std::string& fileName);

	virtual void applyInPlace(cv::Mat& image) override;

	virtual cv::Mat apply(const cv::Mat& image) override;

private:
	HaarDetector faceDetector, eyeDetector;
	//cv::CascadeClassifier faceClassifier, eyeClassifier;
	cv::Mat sunglasses;
};	// SunglassesFilter

SunglassesFilter::SunglassesFilter(const std::string& fileName)
	: faceDetector("./haarcascades/haarcascade_frontalface_default.xml")
	, eyeDetector("./haarcascades/haarcascade_eye.xml")
	, sunglasses(cv::imread(fileName, cv::IMREAD_UNCHANGED))
{
	//CV_Assert(!this->faceClassifier.empty());
	//CV_Assert(!this->eyeClassifier.empty());
	CV_Assert(!this->sunglasses.empty());
	CV_Assert(this->sunglasses.channels() == 4);
}

void SunglassesFilter::applyInPlace(cv::Mat& image)
{
}	// applyInPlace

cv::Mat SunglassesFilter::apply(const cv::Mat& image)
{
	cv::Mat imageCopy = image.clone();
	SunglassesFilter::applyInPlace(imageCopy);
	return imageCopy;
}	// apply


int main(int argc, char* argv[])
{
	try
	{
		SunglassesFilter filter("./images/sunglass.png");

		cv::Mat imInput = cv::imread("./images/mush.jpg", cv::IMREAD_COLOR);
		//cv::Mat imGlasses = cv::imread("./images/sunglass.png", cv::IMREAD_UNCHANGED);
		//CV_Assert(imGlasses.channels() == 4);

		cv::imshow("input", imInput);
		cv::waitKey();

		cv::Mat imOut = filter.apply(imInput);
		cv::imshow("output", imOut);
		cv::waitKey();

		cv::destroyAllWindows();

		return 0;
	}
	catch (const std::exception& e)
	{
		std::cout << e.what() << std::endl;
		return -1;
	}
}