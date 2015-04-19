#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

int main(int, char**)
{
	VideoCapture capture("videos/VID_20150418_151108.mp4");
	Mat frame;

	if(!capture.isOpened()){
		cout << "Error reading video file" << endl;
		return -1;
	}

	while (capture.isOpened()){
		capture >> frame;
		if(frame.empty())
            break;
        imshow("w", frame);
        waitKey(20);
	}

	waitKey(0);

	return 0;
}