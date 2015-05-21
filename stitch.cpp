#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

Mat get_transformation(Mat img_1, Mat img_2){
	int minHessian = 400;
	SurfFeatureDetector detector(minHessian);
	std::vector<KeyPoint> keypoints_1, keypoints_2;
	detector.detect(img_1, keypoints_1);
	detector.detect(img_2, keypoints_2);
	SurfDescriptorExtractor extractor;
	Mat descriptors_1, descriptors_2;
	extractor.compute(img_1, keypoints_1, descriptors_1);
	extractor.compute(img_2, keypoints_2, descriptors_2);
	
}

int main(int, char**)
{
	VideoCapture capture("videos/VID_20150418_151108.mp4");
	Mat frame, prev_frame;

	double frame_width, frame_height, frame_index;

	if(!capture.isOpened()){
		cout << "Error reading video file" << endl;
		return -1;
	} else {
		frame_width = capture.get(CV_CAP_PROP_FRAME_WIDTH);
		frame_height = capture.get(CV_CAP_PROP_FRAME_HEIGHT);
		cout << "frame width = " << frame_width << endl;
		cout << "frame height = " << frame_height << endl;
	}

	while (capture.isOpened()){
		frame_index = capture.get(CV_CAP_PROP_POS_FRAMES);
		// cout << "frame index = " << frame_index << endl;
		if (frame_index > 0){
			prev_frame = frame;
		}
		capture >> frame;
		if(frame.empty())
            break;
        resize(frame, frame, Size(0, 0), 0.5, 0.5, INTER_LINEAR);
        imshow("w", frame);
        waitKey(20);
	}

	waitKey(0);

	return 0;
}