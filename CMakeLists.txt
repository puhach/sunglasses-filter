cmake_minimum_required(VERSION 2.8.12)

PROJECT(SunglassesFilter)
SET(CMAKE_CXX_STANDARD 17)


set(OpenCV_DIR /opt/opencv/4.4.0/installation/lib/cmake/opencv4)

######################## EDIT IF REQUIRED ####################
# ###Uncomment the line below and specify the path to OpenCV directory i.e. the path to the OpenCVConfig.cmake file. Check the examples given below.
#SET(OpenCV_DIR Enter-the-path-of-OpenCV-installation-on-your-system)
#SET(OpenCV_DIR /home/hp/workfolder/OpenCV-Installation/installation/OpenCV-master/lib/cmake/opencv4)


if(MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
endif()


find_package( OpenCV REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS})

add_executable(glassify 
	main.cpp 
	abstractdetector.h 
	abstractdetector.cpp 
	haardetector.h 
	haardetector.cpp 
	proportionaleyedetector.h 
	proportionaleyedetector.cpp 
	abstractimagefilter.h 
	abstractimagefilter.cpp 
	sunglassesfilter.h 
	sunglassesfilter.cpp 
	mediafactory.h 
	mediafactory.cpp 
	mediasource.h 
	mediasink.h 
	imagefilereader.h 
	imagefilereader.cpp 
	imagefilewriter.h 
	imagefilewriter.cpp 
	videofilereader.h 
	videofilereader.cpp 
	videofilewriter.h 
	videofilewriter.cpp 
	webcamreader.h 
	webcamreader.cpp 
	dummywriter.h)

target_link_libraries(glassify ${OpenCV_LIBS} )
  
if(MSVC)
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
  set_property(TARGET glassify PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")
endif()


# Copy haarcascades to the output directory
add_custom_command(TARGET glassify
                   POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy_directory
                   "${PROJECT_SOURCE_DIR}/haarcascades"
                   "$<TARGET_FILE_DIR:glassify>/haarcascades")
