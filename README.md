# Sunglasses Filter

This application lets you try on virtual sunglasses. It uses the Haar-based face detector and two versions of the eye detector:

* *Proportional eye detector* computes eye locations using typical proportions of a frontal face

* *Haar-based eye detector* is almost identical to the Haar-based face detector, but uses different parameters 

![sunglasses filter](./assets/cover.jpg)

## Set Up

It is assumed that OpenCV 4.x, C++17 compiler, and cmake 2.18.12 or newer are installed on the system.

### Specify OpenCV_DIR in CMakeLists

Open CMakeLists.txt and set the correct OpenCV directory in the following line:

```
SET(OpenCV_DIR /home/hp/workfolder/OpenCV-Installation/installation/OpenCV-master/lib/cmake/opencv4)
```

Depending on the platform and the way OpenCV was installed, it may be needed to provide the path to cmake files explicitly. On my KUbuntu 20.04 after building OpenCV 4.4.0 from sources the working `OpenCV_DIR` looks like <OpenCV installation path>/lib/cmake/opencv4. On Windows 8.1 after installing a binary distribution of OpenCV 4.2.0 it is C:\OpenCV\build.


### Build the Project

In the root of the project folder create the `build` directory unless it already exists. Then from the terminal run the following:

```
cd build
cmake ..
```

This should generate the build files. When it's done, compile the code:

```
cmake --build . --config release
```

Since Haar cascade classifiers are necessary for the program to run, the project is configured to automatically copy these files to the output directory upon building completion. But if it fails for some reason, please make sure to place the `haarcascades` folder next to the executable file.


## Usage

The program has to be run from the command line. It takes in the path to the sunglasses image (must have an alpha channel), the reflected image, and several optional parameters: 

```
glassify --sunglasses=<sunglasses image file>
	 --reflection=<reflection image file>
	 [--input=<input image, video, or a webcam>]
	 [--output=<output file>]
	 [--opacity=<the opacity of the sunglasses (0..1)>]
	 [--reflectivity=<the reflectivity of the sunglasses (0..1)>]
	 [--use_haar_eye_detector=<true or false>]
	 [--eye_scale_factor=<float>]
	 [--eye_min_neighbors=<integer>]
	 [--face_scale_factor=<float>]
	 [--face_min_neighbors=<integer>] 
	 [--help]
```

Parameter    | Meaning 
------------ | --------------------------------------
help, ? | Prints the help message.
sunglasses | The image of sunglasses to overlay with the input.
reflection | The image reflected in the sunglasses.
input | The input image, video, or a webcam (use "cam:`<webcam index>`"). Defaults to "cam:0".
output | If not empty, specifies the output file.
opacity | The opacity of the sunglasses (0..1). The default value is 0.5. 
reflectivity | The reflectivity of the sunglasses (0..1). The default value is 0.4. 
use_haar_eye_detector | If set to true a Haar-based detector will be used instead of a proportional eye detector. By default, it is false. 
eye_scale_factor | The scale factor for the Haar-based eye detector. Defaults to 1.1.
eye_min_neighbors | Specifies how many neighbors each candidate should have to be retained by a Haar-based eye detector. By default, it is 3.
face_scale_factor | The scale factor for the Haar-based face detector. Defaults to 1.1.
face_min_neighbors | Specifies how many neighbors each candidate should have to be retained by a Haar-based face detector. By default, it is 15.



Sample usage (linux):
```
./glassify --sunglasses=../images/sunglass.png --reflection=../images/lake.jpg
```

This will read the input frames from the web camera and write no ouput, just display the results. 

It is possible to use an image or a video as an input and write the output to the file:
```
./glassify --sunglasses=../images/sunglass.png --reflection=../images/drops.jpg --input=../images/musk.jpg --output=./out.jpg --opacity=0.3 --reflectivity=0.5
```

This will read the input image, overlay the sunglasses, and save the result to the specified output file.