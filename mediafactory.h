#ifndef MEDIAFACTORY_H
#define MEDIAFACTORY_H


#include <opencv2/core.hpp>

#include <string>
#include <memory>
#include <set>

class MediaSource;
class MediaSink;


// A factory class for constructing readers and writers of media data

class MediaFactory
{
public:
	static std::unique_ptr<MediaSource> createReader(const std::string& input, bool loop = false);
	static std::unique_ptr<MediaSink> createWriter(const std::string& output, cv::Size frameSize, double fps);

private:
	static const std::set<std::string> images;
	static const std::set<std::string> video;

	static std::string getFileExtension(const std::string& inputFile);

	static int getWebCamIndex(const std::string &input);
};	// MediaFactory


#endif	// MEDIAFACTORY_H