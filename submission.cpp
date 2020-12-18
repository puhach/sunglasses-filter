
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

	cv::Mat apply(const cv::Mat& image);	// always allocates a new matrix to store the output

	void apply(const cv::Mat& image, cv::Mat& out);		// may be useful if the output matrix of the matching type has already been allocated

	virtual void applyInPlace(cv::Mat& image) = 0;	// stores the result into the same matrix as the input

	// TODO: implement cloning
};	// ImageFilter


cv::Mat ImageFilter::apply(const cv::Mat& image)
{
	cv::Mat imageCopy = image.clone();
	applyInPlace(imageCopy);	// virtual call
	return imageCopy;
}

void ImageFilter::apply(const cv::Mat& image, cv::Mat& out)
{
	image.copyTo(out);
	applyInPlace(out);
}



class Detector
{
public:
	virtual ~Detector() = default;

	//virtual void detect(const cv::Mat &image, std::vector<cv::Mat>& objs) = 0;
	/*virtual*/ void detect(const cv::Mat& image, std::vector<cv::Mat>& objs);

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
	SunglassesFilter(const std::string& sunglassesFile, const std::string &reflectionFile, float opacity=0.5f, float reflectivity=0.4f,
		std::unique_ptr<Detector> eyeDetector = std::make_unique<HaarDetector>("./haarcascades/haarcascade_eye.xml"),
		std::unique_ptr<Detector> faceDetector = std::make_unique<HaarDetector>("./haarcascades/haarcascade_frontalface_default.xml", 1.1, 15));

	// TODO: implement copy/move semantics

	virtual void applyInPlace(cv::Mat& image) override;

	//virtual cv::Mat apply(const cv::Mat& image) override;

private:

	void fitSunglasses(cv::Mat &face, const cv::Rect &eyeRegion);

	std::unique_ptr<Detector> eyeDetector, faceDetector;
	float opacity, reflectivity;
	//cv::Mat sunglasses, reflection;		
	//cv::Mat4b sunglasses4B;
	//cv::Mat1b reflection1B;
	cv::Mat4f sunglasses4F;
	cv::Mat1f reflection1F;
};	// SunglassesFilter

SunglassesFilter::SunglassesFilter(const std::string& sunglassesFile, const std::string &reflectionFile
	, float opacity, float reflectivity, std::unique_ptr<Detector> eyeDetector, std::unique_ptr<Detector> faceDetector)	
	: eyeDetector(eyeDetector ? std::move(eyeDetector) : throw std::invalid_argument("The eye detector is a null pointer."))
	, faceDetector(faceDetector ? std::move(faceDetector) : throw std::invalid_argument("The face detector object is a null pointer."))
	, opacity(opacity>=0 && opacity<=1 ? opacity : throw std::invalid_argument("The value of opacity must be in range 0..1."))
	, reflectivity(reflectivity>=0 && reflectivity<=1 ? reflectivity : throw std::invalid_argument("The value of reflectivity must be in range 0..1."))
	//, sunglasses4B(cv::imread(sunglassesFile, cv::IMREAD_UNCHANGED))
	//, reflection1B(cv::imread(reflectionFile, cv::IMREAD_GRAYSCALE))
{
	// Read the image of sunglasses preserving the alpha channel
	cv::Mat sunglasses = cv::imread(sunglassesFile, cv::IMREAD_UNCHANGED);
	CV_Assert(!sunglasses.empty());
	CV_Assert(sunglasses.channels() == 4);
	
	// Read the reflection image in the grayscale mode
	cv::Mat reflection = cv::imread(reflectionFile, cv::IMREAD_GRAYSCALE);
	CV_Assert(!reflection.empty());
	CV_Assert(reflection.channels() == 1);

	// Convert the sunglasses and reflection matrices to float just once 
	sunglasses.convertTo(this->sunglasses4F, CV_32F, 1 / 255.0);
	reflection.convertTo(this->reflection1F, CV_32F, 1 / 255.0);

}	// constructor

void SunglassesFilter::applyInPlace(cv::Mat& image)
{
	std::vector<cv::Mat> faces;
	this->faceDetector->detect(image, faces);

	for (/*const*/ cv::Mat& face : faces)
	{
		cv::rectangle(face, cv::Rect(0, 0, face.cols-1, face.rows-1), cv::Scalar(0,0,255));

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

		
		// There must be two eyes, otherwise we just skip this face
		if (eyesEnd - eyeRects.begin() < 2)
			continue;

		//if (eyeRects[0].y > eyeRects[1].y && eyeRects[0].y < eyeRects[1].y + eyeRects[1].height 
		//	|| eyeRects[0].y + eyeRects[0].height > eyeRects[1].y && eyeRects[0].y + eyeRects[0].height < eyeRects[1].y + eyeRects[1].height)

		// Eye rectangles must be roughly on the same level
		if (eyeRects[0].y + eyeRects[0].height < eyeRects[1].y || eyeRects[0].y > eyeRects[1].y + eyeRects[1].height)
			continue;

		fitSunglasses(face, eyeRects[0] | eyeRects[1]);	// minimum area rectangle containing both eye rectangles

	}	// faces
}	// applyInPlace

//cv::Mat SunglassesFilter::apply(const cv::Mat& image)
//{
//	cv::Mat imageCopy = image.clone();
//	SunglassesFilter::applyInPlace(imageCopy);
//	return imageCopy;
//}	// apply

void SunglassesFilter::fitSunglasses(cv::Mat& face, const cv::Rect& eyeRegion)
{
	CV_Assert(face.channels() == 3);
	CV_Assert(eyeRegion.width <= face.cols && eyeRegion.height <= face.rows);
	
	
	// As sunglasses are larger than the eye region, to fit them properly we need to know the size of the face at the level of eyes. 
	// The eyes are supposed to be horizontally centered in the face, so the real face size can't be larger than the eye region width + 
	// two minimal distances from the eyes to the face boundaries. By finding the ratio of the sunglasses width and the face width, 
	// we can compute the scaling factor to resize the sunglasses image preserving the aspect ratio. Although it's unlikely to happen, 
	// but the resized height of the sunglasses must not go beyond the face height.
	double fx = (2.0 * std::min(eyeRegion.x, face.cols - eyeRegion.x - eyeRegion.width) + eyeRegion.width) / this->sunglasses4F.cols;
	double fy = (2.0 * std::min(eyeRegion.y, face.rows - eyeRegion.y - eyeRegion.height) + eyeRegion.height) / this->sunglasses4F.rows;
	double f = std::min(fx, fy);	// make sure glasses do not exceed the face boundaries

	// Resize the image of sunglasses preserving the aspect ratio
	cv::Mat4f sunglassesResized4F;
	cv::resize(this->sunglasses4F, sunglassesResized4F, cv::Size(), f, f);
	CV_Assert(sunglassesResized4F.cols > 0 && sunglassesResized4F.rows > 0);

		
	// Having resized the image of sunglasses, we need to extend the eye region to match the size of the glasses
	cv::Mat3b sunglassesROI3B = face(eyeRegion);
	int dx = sunglassesResized4F.cols - eyeRegion.width;
	int dy = sunglassesResized4F.rows - eyeRegion.height;
	sunglassesROI3B.adjustROI(dy/2, dy/2+dy%2, dx/2, dx/2+dx%2);	// boundaries of the adjusted ROI are constrained by boundaries of the parent matrix
	CV_Assert(sunglassesROI3B.size() == sunglassesResized4F.size());
	
	// Scale the pixel values to 0..1
	cv::Mat3f sunglassesROI3F;
	sunglassesROI3B.convertTo(sunglassesROI3F, CV_32F, 1 / 255.0);


	// Extract BGRA channels from the image of sunglasses
	std::vector<cv::Mat1f> channels;
	cv::split(sunglassesResized4F, channels);

	// Obtain the alpha mask 
	cv::Mat1f mask1F = channels[3];


	// Resize the reflection image to match the size of sunglasses
	cv::Mat1f reflectionResized1F;
	cv::resize(this->reflection1F, reflectionResized1F, mask1F.size());

	// Make the reflection semi-transparent
	cv::multiply(reflectionResized1F, mask1F, reflectionResized1F, this->reflectivity);

	// The non-reflected part comes from the image of sunglasses
	for (int i = 0; i < 3; ++i)
	{
		cv::multiply(channels[i], mask1F, channels[i], 1.0-this->reflectivity);
		channels[i] += reflectionResized1F;
	}

	// Remove the alpha channel to create a 3-channel image of sunglasses (so as to match the ROI)
	channels.pop_back();
	cv::Mat3f sunglassesResized3F;
	cv::merge(channels, sunglassesResized3F);


	// Create a 3-channel mask to match the ROI
	cv::Mat3f mask3F;
	cv::merge(std::vector<cv::Mat1f>{ mask1F, mask1F, mask1F }, mask3F);

	
	// Overlay the face with the sunglasses
	// 1) sunglasses' = sunglasses*mask*opacity
	// 2) face' = face*(1 - mask*opacity) = face - face*mask*opacity
	// 3) result = face' + sunglasses' = face - face*mask*opacity + sunglasses*mask*opacity = face + mask*opacity*(sunglasses-face)
	sunglassesResized3F -= sunglassesROI3F;
	cv::multiply(sunglassesResized3F, mask3F, sunglassesResized3F, this->opacity);
	sunglassesROI3F += sunglassesResized3F;
	sunglassesROI3F.convertTo(sunglassesROI3B, CV_8U, 255);
	

	// The following code can be used to make the border of the sunglasses opaque
	/*
	cv::Mat1f maskEroded1F;
	cv::erode(mask1F, maskEroded1F, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));

	cv::Mat3f maskEroded3F;
	cv::merge(std::vector<cv::Mat1f>{maskEroded1F, maskEroded1F, maskEroded1F}, maskEroded3F);
	cv::Mat3f borderMask3F = mask3F - maskEroded3F;

	sunglassesROI3F = sunglassesResized3F.mul(borderMask3F + maskEroded3F * opacity)
		+ sunglassesROI3F.mul(cv::Scalar::all(1) - mask3F + maskEroded3F * (1.0-opacity));

	sunglassesROI3F.convertTo(sunglassesROI3B, CV_8U, 255);*/
}	// fitSunglasses


int main(int argc, char* argv[])
{
	try
	{
		//SunglassesFilter filter("./images/sunglass.png");
		//auto faceDetector = std::make_unique<HaarDetector>("./haarcascades/haarcascade_frontalface_default.xml");
		auto eyeDetector = std::make_unique<ProportionalEyeDetector>();
		//auto eyeDetector = std::make_unique<HaarDetector>("./haarcascades/haarcascade_eye.xml");
		SunglassesFilter filter("./images/sunglass.png", "./images/lake.jpg", 0.5f, 0.4f, std::move(eyeDetector));

		
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