#ifndef MEDIASINK_H
#define MEDIASINK_H



#include <opencv2/core.hpp>

#include <string>




enum class MediaSinkType
{
	ImageFile,
	VideoFile,
	Dummy
};



// A common ancestor for various media sinks

class MediaSink
{
public:

	MediaSinkType getMediaType() const noexcept { return this->mediaType; }

	const std::string& getMediaPath() const noexcept { return this->mediaPath; }

	virtual void write(const cv::Mat& frame) = 0;

	virtual ~MediaSink() = default;

protected:
	MediaSink(MediaSinkType mediaType, const std::string& mediaPath)
		: mediaType(mediaType)
		, mediaPath(mediaPath) {}	// std::string's copy constructor is not noexcept 

	MediaSink(const MediaSink&) = delete;
	MediaSink(MediaSink&&) = delete;

	MediaSink& operator = (const MediaSink&) = delete;
	MediaSink& operator = (MediaSink&&) = delete;

private:
	MediaSinkType mediaType;
	std::string mediaPath;
};	// MediaSink


#endif	// MEDIASINK_H