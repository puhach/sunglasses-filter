#include "mediafactory.h"
#include "imagefilereader.h"
#include "imagefilewriter.h"
#include "videofilereader.h"
#include "videofilewriter.h"
#include "webcamreader.h"
#include "dummywriter.h"

#include <filesystem>
#include <regex>



const std::set<std::string> MediaFactory::images{ ".jpg", ".jpeg", ".png", ".bmp" };
const std::set<std::string> MediaFactory::video{ ".mp4", ".avi" };



std::string MediaFactory::getFileExtension(const std::string& fileName)
{
	return std::filesystem::path(fileName).extension().string();
}

int MediaFactory::getWebCamIndex(const std::string& input)
{
	// Match the "cam" prefix optionally followed by the colon and the camera index
	// https://en.cppreference.com/w/cpp/regex/regex_match
	// https://stackoverflow.com/questions/18633334/regex-optional-group/18633467
	std::regex re("^cam(?::(\\d+))?$", std::regex::icase);
	std::smatch matches;
	if (std::regex_match(input, matches, re))	
	{
		if (matches.size() == 2)	// first submatch is the whole string, the next submatch is the first parenthesized expression
		{
			if (matches[1].matched)	// the second submatch is in the non-capturing group, hence it may not be matched
			{
				std::size_t pos;
				int camIndex = std::stoi(matches[1].str(), &pos);
				if (pos == matches[1].length())
					return camIndex;
			}
			else return 0;	// optional :<camera index> not matched, return the default camera
		}
	}
		
	return -1;
}

std::unique_ptr<MediaSource> MediaFactory::createReader(const std::string& input, bool loop)
{	
	std::string ext = getFileExtension(input);
	std::transform(ext.begin(), ext.end(), ext.begin(), [](char c) { return std::tolower(c); });

	if (images.find(ext) != images.end())
		return std::make_unique<ImageFileReader>(input, loop);
	else if (video.find(ext) != video.end())
		return std::make_unique<VideoFileReader>(input, loop);
	else if (int camIndex = getWebCamIndex(input); camIndex >= 0)
		return std::make_unique<WebcamReader>(camIndex);	
	else
	{		
		// TODO: may handle other input types here, e.g. URLs

		throw std::runtime_error(std::string("Input file type is not supported: ").append(input));
	}
}	// createReader


std::unique_ptr<MediaSink> MediaFactory::createWriter(const std::string& output, cv::Size frameSize, double fps)
{
	if (output.empty())
		return std::make_unique<DummyWriter>();

	std::string ext = MediaFactory::getFileExtension(output);
	if (images.find(ext) != images.end())
		return std::make_unique<ImageFileWriter>(output, frameSize);
	else if (video.find(ext) != video.end())
		// Help the compiler to deduce the argument types which are passed to the constructor
		return std::make_unique<VideoFileWriter, const std::string&, cv::Size, const char(&)[4], double>(output, std::move(frameSize), { 'm','p','4','v' }, std::move(fps));
	else
	{
		// TODO: consider adding other sinks

		throw std::runtime_error(std::string("Output file type is not supported: ").append(ext));
	}
}	// createWriter

