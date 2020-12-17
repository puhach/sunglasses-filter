
#include <iostream>
#include <exception>
#include <memory>
#include <vector>

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

	virtual void detect(const cv::Mat &image, std::vector<cv::Mat>& objs) = 0;

	virtual void detect(const cv::Mat &image, std::vector<cv::Rect>& rects) = 0;
};	// Detector

class HaarDetector : public Detector
{
public:

	HaarDetector(const cv::String &fileName, double scaleFactor = 1.1, int minNeighbors = 3, 
		int flags = 0, cv::Size minSize = cv::Size(), cv::Size maxSize = cv::Size());

	// TODO: define getters and setters

	virtual void detect(const cv::Mat &image, std::vector<cv::Mat>& objs) override;

	virtual void detect(const cv::Mat &image, std::vector<cv::Rect>& rects) override;

private:
	cv::CascadeClassifier cascadeClassifier;
	double scaleFactor;
	int minNeighbors;
	int flags;
	cv::Size minSize, maxSize;
};	// HaarFaceDetector

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

void HaarDetector::detect(const cv::Mat &image, std::vector<cv::Mat>& objs)
{
	std::vector<cv::Rect> rects;
	HaarDetector::detect(image, rects);

	objs.reserve(rects.size());
	for (const cv::Rect& r : rects)
	{
		objs.push_back(image(r));
	}
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


class SunglassesFilter : public ImageFilter
{
public:
	SunglassesFilter(const std::string& fileName,
		std::unique_ptr<Detector> faceDetector = std::make_unique<HaarDetector>("./haarcascades/haarcascade_frontalface_default.xml"),
		std::unique_ptr<Detector> eyeDetector = std::make_unique<HaarDetector>("./haarcascades/haarcascade_eye.xml"));

	// TODO: implement copy/move semantics

	virtual void applyInPlace(cv::Mat& image) override;

	virtual cv::Mat apply(const cv::Mat& image) override;

private:
	std::unique_ptr<Detector> faceDetector, eyeDetector;
	//cv::CascadeClassifier faceClassifier, eyeClassifier;
	cv::Mat sunglasses;
};	// SunglassesFilter

SunglassesFilter::SunglassesFilter(const std::string& fileName, std::unique_ptr<Detector> faceDetector, std::unique_ptr<Detector> eyeDetector)
	: faceDetector(faceDetector ? std::move(faceDetector) : throw std::runtime_error("The face detector object is a null pointer."))
	, eyeDetector(eyeDetector ? std::move(eyeDetector) : throw std::runtime_error("The eye detector is a null pointer."))
	//, eyeDetector("./haarcascades/haarcascade_eye_tree_eyeglasses.xml")
	, sunglasses(cv::imread(fileName, cv::IMREAD_UNCHANGED))
{
	//CV_Assert(!this->faceClassifier.empty());
	//CV_Assert(!this->eyeClassifier.empty());
	CV_Assert(!this->sunglasses.empty());
	CV_Assert(this->sunglasses.channels() == 4);
}

void SunglassesFilter::applyInPlace(cv::Mat& image)
{
	std::vector<cv::Mat> faces;
	this->faceDetector->detect(image, faces);

	for (/*const*/ cv::Mat& face : faces)
	{
		std::vector<cv::Rect> eyeRects;
		this->eyeDetector->detect(face, eyeRects);

		/*
		if (eyeRects.empty())
			continue;

		int x1 = face.cols, x2 = 0, y1 = face.rows, y2 = 0;
		for (const cv::Rect& eyeRect : eyeRects)
		{
			x1 = std::min(x1, eyeRect.x);
			x2 = std::max(x2, eyeRect.x);
			y1 = std::min(y1, eyeRect.y);
			y2 = std::max(y2, eyeRect.y);

			cv::rectangle(face, eyeRect, cv::Scalar(255,0,0));
			cv::imshow("face", face);
			cv::waitKey(10);
		}
		*/

		if (eyeRects.size() < 2)
			continue;

		//if (eyeRects[0].y > eyeRects[1].y && eyeRects[0].y < eyeRects[1].y + eyeRects[1].height 
		//	|| eyeRects[0].y + eyeRects[0].height > eyeRects[1].y && eyeRects[0].y + eyeRects[0].height < eyeRects[1].y + eyeRects[1].height)

		// Eye rectangles must be roughly on the same level
		if (eyeRects[0].y + eyeRects[0].height < eyeRects[1].y || eyeRects[0].y > eyeRects[1].y + eyeRects[1].height)
			continue;

		int x1 = std::min(eyeRects[0].x, eyeRects[1].x);
		int x2 = std::max(eyeRects[0].x + eyeRects[0].width, eyeRects[1].x + eyeRects[1].width);
		int y1 = std::min(eyeRects[0].y, eyeRects[1].y);
		int y2 = std::max(eyeRects[0].y+eyeRects[0].height, eyeRects[1].y+eyeRects[1].height);

		cv::rectangle(face, cv::Rect(x1, y1, x2 - x1, y2 - y1), cv::Scalar(0, 255, 0));
		//cv::imshow("eyes", image);
		//cv::waitKey(10);
	}	// faces
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

		//cv::Mat imInput = cv::imread("./images/musk.jpg", cv::IMREAD_COLOR);
		//cv::Mat imGlasses = cv::imread("./images/sunglass.png", cv::IMREAD_UNCHANGED);
		//CV_Assert(imGlasses.channels() == 4);

		/*cv::imshow("input", imInput);
		cv::waitKey();

		cv::Mat imOut = filter.apply(imInput);
		cv::imshow("output", imOut);
		cv::waitKey();*/

		cv::VideoCapture cap(0);
		while (cap.isOpened())
		{
			cv::Mat frame;
			cap >> frame;
			//cv::imshow("input", imInput);
			//cv::waitKey();
			cv::Mat out = filter.apply(frame);
			cv::imshow("test", out);
			cv::waitKey(10);
		}

		cv::destroyAllWindows();

		return 0;
	}
	catch (const std::exception& e)
	{
		std::cout << e.what() << std::endl;
		return -1;
	}
}