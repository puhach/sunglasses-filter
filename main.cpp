#include "mediafactory.h"
#include "mediasource.h"
#include "mediasink.h"
#include "haardetector.h"
#include "proportionaleyedetector.h"
#include "sunglassesfilter.h"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>




void printUsage()
{
	std::cout << "Usage: glassify.exe [-h]"
				 " --sunglasses=<sunglasses image file>" 
				 " --reflection=<reflection image file>"
				 " [--input=<input image, video, or a webcam>]"
				 " [--output=<output file>]"
				 " [--opacity=<the opacity of the sunglasses (0..1)>]"
				 " [--reflectivity=<the reflectivity of the sunglasses (0..1)>]"
				 " [--use_haar_eye_detector=<true or false>]"
				 " [--eye_scale_factor=<float>]"
				 " [--eye_min_neighbors=<integer>]"
				 " [--face_scale_factor=<float>]"
				 " [--face_min_neighbors=<integer>]" << std::endl;
}

int main(int argc, char* argv[])
{
	try
	{
		
		static const cv::String keys =
			"{help h usage ?        |                     | Print the help message  }"			
			"{sunglasses            |<none>               | The image of sunglasses to overlay with the input }"
			"{reflection            |<none>               | The reflection image for sunglasses }"
			"{input                 |cam:0                | The input image, video, or a webcam  }"
			"{output                |                     | If not empty, specifies the output file }"
			"{opacity               |0.5                  | The opacity of the sunglasses (0..1) }"
			"{reflectivity          |0.4                  | The reflectivity of the sunglasses (0..1) }"
			"{use_haar_eye_detector |false                | If set to true a Haar-based detector will be used instead of a proportional eye detector }"
			"{eye_scale_factor      |1.1                  | The scale factor for the Haar-based eye detector }"
			"{eye_min_neighbors     |3                    | How many neighbors each candidate should have to be retained by a Haar-based eye detector }"
			"{face_scale_factor     |1.1                  | The scale factor for the Haar-based face detector }"
			"{face_min_neighbors    |15                   | How many neighbors each candidate should have to be retained by a Haar-based face detector }";
		
		cv::CommandLineParser parser(argc, argv, keys);
		parser.about("Glassify\n(c) Yaroslav Pugach");

		if (parser.has("help"))
		{
			printUsage();
			return 0;
		}

		std::string sunglassesFile = parser.get<std::string>("sunglasses");
		std::string reflectionFile = parser.get<std::string>("reflection");
		std::string input = parser.get<std::string>("input");
		std::string output = parser.get<std::string>("output");
		float opacity = parser.get<float>("opacity");
		float reflectivity = parser.get<float>("reflectivity");
		bool useHaarEyeDetector = parser.get<bool>("use_haar_eye_detector");
		double eyeScaleFactor = parser.get<double>("eye_scale_factor");
		int eyeMinNeighbors = parser.get<int>("eye_min_neighbors");
		double faceScaleFactor = parser.get<double>("face_scale_factor");
		int faceMinNeighbors = parser.get<int>("face_min_neighbors");

		if (!parser.check())
		{
			parser.printErrors();
			printUsage();
			return -1;
		}

		auto reader = MediaFactory::createReader(input);
		auto writer = MediaFactory::createWriter(output, reader->getFrameSize(), 10);
		
		auto faceDetector = std::make_unique<HaarDetector>("./haarcascades/haarcascade_frontalface_default.xml", faceScaleFactor, faceMinNeighbors);
		auto eyeDetector = useHaarEyeDetector ? std::make_unique<HaarDetector>("./haarcascades/haarcascade_eye.xml", eyeScaleFactor, eyeMinNeighbors)
											  : std::unique_ptr<AbstractDetector>(new ProportionalEyeDetector);
				
		SunglassesFilter filter(sunglassesFile, reflectionFile, opacity, reflectivity, std::move(eyeDetector), std::move(faceDetector));
						
		cv::Mat frame;
		while (reader->readNext(frame))
		{
			filter.applyInPlace(frame);
			cv::imshow("Glassify", frame);
			writer->write(frame);
			int key = cv::waitKey(reader->getMediaType() == MediaSourceType::ImageFile ? 0 : 10);
			if ((key & 0xFF) == 27)
				break;
		}

		cv::destroyAllWindows();

		return 0;
	}
	catch (const std::exception& e)
	{
		std::cerr << e.what() << std::endl;
		return -1;
	}
}