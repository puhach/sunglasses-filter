#ifndef SUNGLASSESFILTER_H
#define SUNGLASSESFILTER_H



#include "abstractimagefilter.h"
#include "haardetector.h"

#include <opencv2/core.hpp>

#include <string>
#include <memory>

class AbstractDetector;


// An image filter for adding virtual sunglasses


class SunglassesFilter: public AbstractImageFilter
{
public:
	SunglassesFilter(const std::string& sunglassesFile, const std::string &reflectionFile, float opacity=0.5f, float reflectivity=0.4f,
		std::unique_ptr<AbstractDetector> eyeDetector = std::make_unique<HaarDetector>("./haarcascades/haarcascade_eye.xml"),
		std::unique_ptr<AbstractDetector> faceDetector = std::make_unique<HaarDetector>("./haarcascades/haarcascade_frontalface_default.xml", 1.1, 15));

	~SunglassesFilter() = default;

	float getOpacity() const noexcept { return this->opacity; }
	void setOpacity(float opacity) noexcept { this->opacity = opacity; }

	float getReflectivity() const noexcept { return this->reflectivity; }
	void setReflectivity(float reflectivity) noexcept { this->reflectivity = reflectivity; }

	//AbstractDetector* getEyeDetector() const noexcept { return this->eyeDetector.get(); }
	void setEyeDetector(std::unique_ptr<AbstractDetector> eyeDetector) noexcept { this->eyeDetector = std::move(eyeDetector); }

	//AbstractDetector* getFaceDetector() const noexcept { return this->faceDetector.get(); }
	void setFaceDetector(std::unique_ptr<AbstractDetector> faceDetector) noexcept { this->faceDetector = std::move(faceDetector); }

	virtual void applyInPlace(cv::Mat& image) const override;

	virtual std::unique_ptr<AbstractImageFilter> clone() const& override;
	virtual std::unique_ptr<AbstractImageFilter> clone() && override;	

protected:
	SunglassesFilter(const SunglassesFilter& other);
	SunglassesFilter(SunglassesFilter&& other);

private:

	void fitSunglasses(cv::Mat &face, const cv::Rect &eyeRegion) const;

	std::unique_ptr<AbstractDetector> eyeDetector, faceDetector;
	float opacity, reflectivity;
	cv::Mat4f sunglasses4F;
	cv::Mat1f reflection1F;
};	// SunglassesFilter


#endif 	// SUNGLASSESFILTER_H