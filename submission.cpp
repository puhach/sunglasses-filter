
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

	//virtual void detect(const cv::Mat &image, std::vector<cv::Mat>& objs) = 0;
	virtual void detect(const cv::Mat& image, std::vector<cv::Mat>& objs);

	virtual void detect(const cv::Mat &image, std::vector<cv::Rect>& rects) = 0;
};	// Detector

void Detector::detect(const cv::Mat& image, std::vector<cv::Mat>& objs)
{
	std::vector<cv::Rect> rects;
	detect(image, rects);	// virtual

	objs.reserve(rects.size());
	for (const cv::Rect& r : rects)
	{
		objs.push_back(image(r));
	}
}	// detect


class ProportionalEyeDetector : public Detector
{
public:

	//virtual void detect(const cv::Mat& image, std::vector<cv::Mat>& objs) override;

	virtual void detect(const cv::Mat &face, std::vector<cv::Rect> &rects) override;
};	// ProportionalEyeDetector


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


class HaarDetector : public Detector
{
public:

	HaarDetector(const cv::String &fileName, double scaleFactor = 1.1, int minNeighbors = 3, 
		int flags = 0, cv::Size minSize = cv::Size(), cv::Size maxSize = cv::Size());

	// TODO: define getters and setters

	//virtual void detect(const cv::Mat &image, std::vector<cv::Mat>& objs) override;

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

//void HaarDetector::detect(const cv::Mat &image, std::vector<cv::Mat>& objs)
//{
//	std::vector<cv::Rect> rects;
//	HaarDetector::detect(image, rects);
//
//	objs.reserve(rects.size());
//	for (const cv::Rect& r : rects)
//	{
//		objs.push_back(image(r));
//	}
//}

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
		std::unique_ptr<Detector> eyeDetector = std::make_unique<HaarDetector>("./haarcascades/haarcascade_eye.xml"),
		std::unique_ptr<Detector> faceDetector = std::make_unique<HaarDetector>("./haarcascades/haarcascade_frontalface_default.xml"));

	// TODO: implement copy/move semantics

	virtual void applyInPlace(cv::Mat& image) override;

	virtual cv::Mat apply(const cv::Mat& image) override;

private:

	void fitSunglasses(cv::Mat &face, const cv::Rect &eyeRegion);

	std::unique_ptr<Detector> eyeDetector, faceDetector;
	//cv::CascadeClassifier faceClassifier, eyeClassifier;
	cv::Mat sunglasses;
};	// SunglassesFilter

SunglassesFilter::SunglassesFilter(const std::string& fileName, std::unique_ptr<Detector> eyeDetector, std::unique_ptr<Detector> faceDetector)
	: eyeDetector(eyeDetector ? std::move(eyeDetector) : throw std::runtime_error("The eye detector is a null pointer."))
	, faceDetector(faceDetector ? std::move(faceDetector) : throw std::runtime_error("The face detector object is a null pointer."))
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
		//if (eyeRects.empty())
		//	continue;

		////int x1 = face.cols, x2 = 0, y1 = face.rows, y2 = 0;
		for (const cv::Rect& eyeRect : eyeRects)
		{
			//x1 = std::min(x1, eyeRect.x);
			//x2 = std::max(x2, eyeRect.x);
			//y1 = std::min(y1, eyeRect.y);
			//y2 = std::max(y2, eyeRect.y);

			cv::rectangle(face, eyeRect, cv::Scalar(255,0,0));
			//cv::imshow("face", face);
			//cv::waitKey(10);
		}
		*/

		// Eyes are expected to be in the top part of the face
		auto eyesEnd = std::remove_if(eyeRects.begin(), eyeRects.end(), [&face](const cv::Rect& r) {
				return r.y > face.rows / 2;
			});

		
		//if (eyeRects.size() < 2)
		if (eyesEnd - eyeRects.begin() < 2)
			continue;

		//if (eyeRects[0].y > eyeRects[1].y && eyeRects[0].y < eyeRects[1].y + eyeRects[1].height 
		//	|| eyeRects[0].y + eyeRects[0].height > eyeRects[1].y && eyeRects[0].y + eyeRects[0].height < eyeRects[1].y + eyeRects[1].height)

		// Eye rectangles must be roughly on the same level
		if (eyeRects[0].y + eyeRects[0].height < eyeRects[1].y || eyeRects[0].y > eyeRects[1].y + eyeRects[1].height)
			continue;

		/*
		cv::Rect eyeRegion = eyeRects[0] | eyeRects[1];	// minimum area rectangle containing both eye rectangles
		
		// Compute the sunglasses region from the eye region
		int sunglassesHeight = static_cast<int>(eyeRegion.height / (eyeRegion.width + 1e-5) * face.cols);
		cv::Rect sunglassesRegion(0, eyeRegion.y+(eyeRegion.height-sunglassesHeight)/2, face.cols, sunglassesHeight);
		*/

		fitSunglasses(face, eyeRects[0] | eyeRects[1]);	// minimum area rectangle containing both eye rectangles

		/*
		//cv::rectangle(face, cv::Rect(x1, y1, x2 - x1, y2 - y1), cv::Scalar(0, 255, 0));
		cv::rectangle(face, sunglassesRegion, cv::Scalar(0, 255, 0));
		cv::imshow("face", face);
		cv::waitKey(10);
		*/
	}	// faces
}	// applyInPlace

cv::Mat SunglassesFilter::apply(const cv::Mat& image)
{
	cv::Mat imageCopy = image.clone();
	SunglassesFilter::applyInPlace(imageCopy);
	return imageCopy;
}	// apply

void SunglassesFilter::fitSunglasses(cv::Mat& face, const cv::Rect& eyeRegion)
{
	CV_Assert(face.channels() == 3);
	CV_Assert(eyeRegion.width <= face.cols && eyeRegion.height <= face.rows);
	
	// TODO: convert sunglasses to float just once in the ctor
	cv::Mat4f sunglassesF;
	this->sunglasses.convertTo(sunglassesF, CV_32F, 1/255.0);


	// Compute the sunglasses region from the eye region
	/*int sunglassesHeight = static_cast<int>(face.cols / (eyeRegion.width + 1e-5) * eyeRegion.height);
	cv::Rect sunglassesRect(0, eyeRegion.y + (eyeRegion.height - sunglassesHeight) / 2, face.cols, sunglassesHeight);
	*/

	// As sunglasses are larger than the eye region, to fit them properly we need to know the size of the face at the level of eyes. 
	// The eyes are supposed to be horizontally centered in the face, so the real face size can't be larger than the eye region width + 
	// two minimal distances to the face boundary. By finding the ratio of the sunglasses width and the face width, we can compute 
	// the scaling factor to resize the sunglasses image preserving the aspect ratio. Although it's unlikely to happen, but the resized 
	// height of the sunglasses must not go beyond the face height.
	double fx = 1.0 * (2.0 * std::min(eyeRegion.x, face.cols - eyeRegion.x - eyeRegion.width) + eyeRegion.width) / this->sunglasses.cols;
	double fy = 1.0 * (2.0 * std::min(eyeRegion.y, face.rows - eyeRegion.y - eyeRegion.height) + eyeRegion.height) / this->sunglasses.rows;
	//double fx = 1.0*face.cols / this->sunglasses.cols;	// TODO: introduce a fit factor
	//double fy = 1.0 * face.rows / this->sunglasses.rows;
	double f = std::min(fx, fy);	// make sure glasses do not exceed the face boundaries

	// Resize the image of sunglasses preserving the aspect ratio
	cv::Mat4f sunglassesResizedF;
	//cv::resize(sunglassesF, sunglassesResizedF, sunglassesRect.size());
	cv::resize(sunglassesF, sunglassesResizedF, cv::Size(), f, f);


	cv::imshow("test", sunglassesResizedF);
	cv::waitKey();


	// Having resized the image of sunglasses, we need to extend the eye region to match the size of the glasses
	//cv::Size dsz = cv::max( sunglassesResizedF.size() - eyeRegion.size(), cv::Size(0,0));
	cv::Point dsz = sunglassesResizedF.size() - eyeRegion.size();
	//cv::Rect sunglassesRect(eyeRegion.x - dsz.width/2, eyeRegion.y-dsz.height/2, sunglassesResizedF.cols, sunglassesResizedF.rows);
	cv::Rect sunglassesRect(eyeRegion.tl() - dsz/2, sunglassesResizedF.size());
	CV_Assert(eyeRegion.x >= 0 && eyeRegion.y >= 0 && eyeRegion.x + eyeRegion.width < face.cols && eyeRegion.y + eyeRegion.height < face.rows);

	cv::Mat3b sunglassesROIB = face(sunglassesRect);
	cv::Mat3f sunglassesROIF;
	sunglassesROIB.convertTo(sunglassesROIF, CV_32F, 1 / 255.0);


	cv::imshow("test", sunglassesROIF);
	cv::waitKey();

	std::vector<cv::Mat1f> channels;
	cv::split(sunglassesResizedF, channels);

	// Obtain the alpha mask 
	cv::Mat1f mask1F = channels[3];
	cv::imshow("test", mask1F);
	cv::waitKey();

}	// fitSunglasses


int main(int argc, char* argv[])
{
	try
	{
		//SunglassesFilter filter("./images/sunglass.png");
		//auto faceDetector = std::make_unique<HaarDetector>("./haarcascades/haarcascade_frontalface_default.xml");
		auto eyeDetector = std::make_unique<ProportionalEyeDetector>();
		SunglassesFilter filter("./images/sunglass.png", std::move(eyeDetector));

		
		cv::Mat imInput = cv::imread("./images/musk.jpg", cv::IMREAD_COLOR);
		//cv::Mat imGlasses = cv::imread("./images/sunglass.png", cv::IMREAD_UNCHANGED);
		//CV_Assert(imGlasses.channels() == 4);

		cv::imshow("input", imInput);
		cv::waitKey();

		cv::Mat imOut = filter.apply(imInput);
		cv::imshow("output", imOut);
		cv::waitKey();
		

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