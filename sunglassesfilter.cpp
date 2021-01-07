#include "sunglassesfilter.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>


SunglassesFilter::SunglassesFilter(const std::string& sunglassesFile, const std::string &reflectionFile
	, float opacity, float reflectivity, std::unique_ptr<AbstractDetector> eyeDetector, std::unique_ptr<AbstractDetector> faceDetector)	
	: eyeDetector(eyeDetector ? std::move(eyeDetector) : throw std::invalid_argument("The eye detector is a null pointer."))
	, faceDetector(faceDetector ? std::move(faceDetector) : throw std::invalid_argument("The face detector object is a null pointer."))
	, opacity(opacity>=0 && opacity<=1 ? opacity : throw std::invalid_argument("The value of opacity must be in range 0..1."))
	, reflectivity(reflectivity>=0 && reflectivity<=1 ? reflectivity : throw std::invalid_argument("The value of reflectivity must be in range 0..1."))
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

SunglassesFilter::SunglassesFilter(const SunglassesFilter& other)
	: eyeDetector(other.eyeDetector->clone())
	, faceDetector(other.faceDetector->clone())
	, opacity(other.opacity)
	, reflectivity(other.reflectivity)
	, sunglasses4F(other.sunglasses4F)	// it is safe not to perform deep cloning here as long as we do not modify 
	, reflection1F(other.reflection1F)	// these matrices
{
}	// copy constructor

SunglassesFilter::SunglassesFilter(SunglassesFilter&& other) 
	: eyeDetector(std::move(other.eyeDetector))
	, faceDetector(std::move(other.faceDetector))
	, opacity(other.opacity)
	, reflectivity(other.reflectivity)
	, sunglasses4F(std::move(other.sunglasses4F))
	, reflection1F(std::move(other.reflection1F))
{
}	// move constructor

std::unique_ptr<AbstractImageFilter> SunglassesFilter::clone() const& 
{
	return std::unique_ptr<SunglassesFilter>(new SunglassesFilter(*this));
}

std::unique_ptr<AbstractImageFilter> SunglassesFilter::clone()&&
{
	return std::unique_ptr<SunglassesFilter>(new SunglassesFilter(std::move(*this)));
}

void SunglassesFilter::applyInPlace(cv::Mat& image) const
{
	std::vector<cv::Mat> faces;
	this->faceDetector->detect(image, faces);

	for (cv::Mat& face : faces)
	{
		std::vector<cv::Rect> eyeRects;
		this->eyeDetector->detect(face, eyeRects);

		// Eyes are expected to be in the top part of the face
		auto eyesEnd = std::remove_if(eyeRects.begin(), eyeRects.end(), [&face](const cv::Rect& r) {
				return r.y > face.rows / 2;
			});
		
		// There must be two eyes, otherwise we just skip this face
		if (eyesEnd - eyeRects.begin() < 2)
			continue;

		// Eye rectangles must be roughly on the same level
		if (eyeRects[0].y + eyeRects[0].height < eyeRects[1].y || eyeRects[0].y > eyeRects[1].y + eyeRects[1].height)
			continue;

		fitSunglasses(face, eyeRects[0] | eyeRects[1]);	// minimum area rectangle containing both eye rectangles
	}	// faces
}	// applyInPlace

void SunglassesFilter::fitSunglasses(cv::Mat& face, const cv::Rect& eyeRegion) const
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

