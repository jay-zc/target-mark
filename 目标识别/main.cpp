#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>

using namespace std;
using namespace cv;

void detectAndDisplay(Mat frame);

CascadeClassifier face_cascade;    //使用级联分类器类加载视频中对象  //face
CascadeClassifier eyes_cascade;            //eyes
CascadeClassifier upperbody_cascade;          //body

int main(int argc, const char** argv)
{
	CommandLineParser parser(argc, argv,
		"{help h||}"
		"{face_cascade|D:/install/opencv3.4.5/opencv/sources/data/haarcascades/haarcascade_frontalface_alt.xml|Path to face cascade.}"
		"{eyes_cascade|D:/install/opencv3.4.5/opencv/sources/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml|Path to eyes cascade.}"
		"{upperbody_cascade|D:/install/opencv3.4.5/opencv/sources/data/haarcascades/haarcascade_upperbody.xml|Path to upperbody cascade.}"
		"{camera|0|Camera device number.}");
	//路径自己改，在你安装OpenCV目录寻找

	parser.about("\nThis program demonstrates using the cv::CascadeClassifier class to detect objects (Face + eyes) in a video stream.\n"
		"You can use Haar or LBP features.\n\n");
	parser.printMessage();

	String face_cascade_name = parser.get<String>("face_cascade");
	String eyes_cascade_name = parser.get<String>("eyes_cascade");
	String upperbody_cascade_name = parser.get<String>("upperbody_cascade");

	//-- 1. Load the cascades
	if (!face_cascade.load(face_cascade_name))
	{
		cout << "--(!)Error loading face cascade\n";
		return -1;
	};
	if (!eyes_cascade.load(eyes_cascade_name))
	{
		cout << "--(!)Error loading eyes cascade\n";
		return -1;
	};
	if (!upperbody_cascade.load(upperbody_cascade_name))
	{
		cout << "--(!)Error loading upperbody cascade\n";
		return -1;
	};

	int camera_device = parser.get<int>("camera");
	VideoCapture capture;
	//-- 2. Read the video stream
	capture.open(camera_device);
	if (!capture.isOpened())
	{
		cout << "--(!)Error opening video capture\n";
		return -1;
	}

	Mat frame;
	while (capture.read(frame))
	{
		if (frame.empty())
		{
			cout << "--(!) No captured frame -- Break!\n";
			break;
		}

		//-- 3. Apply the classifier to the frame
		detect AndDisplay(frame);

		if (waitKey(10) == 27)
		{
			break; // escape
		}
	}

	return 0;
}

void detectAndDisplay(Mat frame)
{
	Mat frame_gray;
	cvtColor(frame, frame_gray, COLOR_BGR2GRAY);     //转灰度图 
	equalizeHist(frame_gray, frame_gray);       //直方图等化：增强图像的对比度

												//-- Detect faces
	std::vector<Rect> faces;
	face_cascade.detectMultiScale(frame_gray, faces);    //在图像中检测不同大小的对象，并将检测到的对象以矩形列表返回，这里检测的是人脸
														 //这个返回值是放在faces里面的，detectMultiScale这个函数是void

	for (size_t i = 0; i < faces.size(); i++)
	{
		Point center(faces[i].x + faces[i].width / 2, \
			faces[i].y + faces[i].height / 2);         //计算到面部的中心点
		ellipse(frame, center, Size(faces[i].width / 2, \
			faces[i].height / 2), 0, 0, 360, Scalar(255, 0, 255), 4);   //绘制圆

		Mat faceROI = frame_gray(faces[i]);          //先确定脸，再去找眼睛

													 //-- In each face, detect eyes
		std::vector<Rect> eyes;
		eyes_cascade.detectMultiScale(faceROI, eyes);

		for (size_t j = 0; j < eyes.size(); j++)
		{
			Point eye_center(faces[i].x + eyes[j].x + eyes[j].width / 2, \
				faces[i].y + eyes[j].y + eyes[j].height / 2);
			int radius = cvRound((eyes[j].width + eyes[j].height)*0.25);
			circle(frame, eye_center, radius, Scalar(255, 0, 0), 4);
		}
	}

	std::vector<Rect> upperbody;
	upperbody_cascade.detectMultiScale(frame_gray, upperbody);
	for (size_t i = 0; i < upperbody.size(); i++)
	{
		Point top_left(upperbody[i].x, upperbody[i].y);
		Point low_right(upperbody[i].x + upperbody[i].width, \
			upperbody[i].y + upperbody[i].height);
		rectangle(frame, top_left, low_right, Scalar(0, 255, 0), 4);
	}

	//-- Show what you got
	imshow("Capture - Face detection", frame);
}