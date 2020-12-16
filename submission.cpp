
#include <iostream>
#include <exception>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

class ImageFilter
{
public:
	virtual ~ImageFilter() = default;

	virtual void applyInPlace(cv::Mat& image) = 0;

	virtual cv::Mat apply(const cv::Mat& image) = 0;
};	// ImageFilter


class SunglassesFilter : public ImageFilter
{
public:
	SunglassesFilter(const std::string& fileName);

	virtual void applyInPlace(cv::Mat& image) override;

	virtual cv::Mat apply(const cv::Mat& image) override;

private:
	std::string fileName;
};	// SunglassesFilter


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

		return 0;
	}
	catch (const std::exception& e)
	{
		std::cout << e.what() << std::endl;
		return -1;
	}
}