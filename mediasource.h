#ifndef MEDIASOURCE_H
#define MEDIASOURCE_H



#include <opencv2/core.hpp>

#include <string>



enum class MediaSourceType	
{
	ImageFile,
	VideoFile,
	Webcam
};



// A common ancestor for various media sources

class MediaSource	// a universal media source
{
public:

	MediaSourceType getMediaType() const noexcept { return this->mediaType; }

	const std::string& getMediaPath() const noexcept { return this->mediaPath; }

	bool isLooped() const noexcept { return this->looped; }

	virtual cv::Size getFrameSize() const = 0;

	virtual bool readNext(cv::Mat& frame) = 0;

	virtual void reset() = 0;

	virtual ~MediaSource() = default;

protected:

	MediaSource(MediaSourceType inputType, const std::string& mediaPath, bool looped)
		: mediaType(inputType)
		, mediaPath(mediaPath)
		, looped(looped) {}

	MediaSource(const MediaSource&) = delete;
	MediaSource(MediaSource&&) = delete;

	MediaSource& operator = (const MediaSource&) = delete;
	MediaSource& operator = (MediaSource&&) = delete;

private:
	MediaSourceType mediaType;
	std::string mediaPath;
	bool looped = false;
};	// MediaSource


#endif	// MEDIASOURCE_H