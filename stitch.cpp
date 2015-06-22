#include "opencv2/opencv.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include <iostream>
#include <stdio.h>

using namespace cv;
using namespace std;

std::vector<KeyPoint> get_keypoints(Mat img){
	int minHessian = 400;
	SurfFeatureDetector detector(minHessian);
	std::vector<KeyPoint> keypoints_1;
	detector.detect(img, keypoints_1);
	return keypoints_1;
} 

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

	FlannBasedMatcher matcher;
	std::vector<DMatch> matches;
	matcher.match(descriptors_2, descriptors_1, matches);

	double max_dist = 0; double min_dist = 100;

	//-- Quick calculation of max and min distances between keypoints
	for( int i = 0; i < descriptors_2.rows; i++ )
	{ double dist = matches[i].distance;
	if( dist < min_dist ) min_dist = dist;
	if( dist > max_dist ) max_dist = dist;
	}

	printf("-- Max dist : %f \n", max_dist );
	printf("-- Min dist : %f \n", min_dist );

	//-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
	std::vector<DMatch> good_matches;

	for( int i = 0; i < descriptors_2.rows; i++ )
	{ if( matches[i].distance < 3*min_dist )
	 { good_matches.push_back( matches[i]); }
	}

	// show the matches from frame to frame
	// for debugging
	if (false){
		Mat img_matches;
		drawMatches( img_2, keypoints_2, img_1, keypoints_1,
		           good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
		           vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

		imshow("matches", img_matches);
	}

	std::vector<Point2f> img_1_forhomography;
	std::vector<Point2f> img_2_forhomography;

	for( int i = 0; i < good_matches.size(); i++ )
	{
	//-- Get the keypoints from the good matches
	img_2_forhomography.push_back( keypoints_2[ good_matches[i].queryIdx ].pt );
	img_1_forhomography.push_back( keypoints_1[ good_matches[i].trainIdx ].pt );
	}

	Mat transform = findHomography(img_2_forhomography, img_1_forhomography, CV_RANSAC);
	return transform;
}

int main(int, char**)
{
	VideoCapture capture("videos/VID_20150418_151108.mp4");
	Mat frame, prev_frame;
	Mat transform;

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

        // only compute the transformation when you have two frames
        // otherwise, prev_frame does not exist and there is a segfault
        if (frame_index > 0){
	        transform = get_transformation(frame, prev_frame);
	        cout << transform << endl;
        }

        // show the keypoints on a frame
        // for debugging
        if (false){
	        std::vector<KeyPoint> keypoints_1 = get_keypoints(frame);
			drawKeypoints(frame, keypoints_1, frame, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
        }

        imshow("w", frame);
        waitKey(20);
	}

	waitKey(0);

	return 0;
}