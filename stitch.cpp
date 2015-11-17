#include "opencv2/opencv.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include <iostream>
#include <cstdlib> 
#include <stdio.h>

using namespace cv;
using namespace std;

#define SAVE_FILES 0
// #define NUM_FRAMES_TO_ANALYZE 10000

// types of blending opportunities
#define BLEND_MEDIAN 1
#define BLEND_PURE 2

std::vector<KeyPoint> get_keypoints(Mat &img){
	int minHessian = 400;
	SurfFeatureDetector detector(minHessian);
	std::vector<KeyPoint> keypoints_1;
	detector.detect(img, keypoints_1);
	return keypoints_1;
} 

// takes in image dimensions and a transformation (homography)
// returns an opencv Rect (x, y, width, height)
// that represents the new bounding box size
cv::Rect transformed_bbox(int imwidth, int imheight, Mat transform){
	// initialize bbox
    cv::Rect bbox;

	// transform the 4 points
    // (0, 0); (0, imheight); (imwidth, 0); (imwidth; imheight);

	float test_x[] = {0, 0, (float)(imwidth), (float)(imwidth)};
    float test_y[] = {0, (float)(imheight), 0, (float)(imheight)};

    Mat test_point(3, 1, CV_32FC1);
    
    for (int index = 0; index < 4; ++index){
        test_point.at<float>(0, 0) = test_x[index];
        test_point.at<float>(0, 1) = test_y[index];
        test_point.at<float>(0, 2) = 1.0f;

        // compute the transformation
        test_point = transform * test_point;

        // homogenize the coordinate
        test_point.at<float>(0, 0) = test_point.at<float>(0, 0) / test_point.at<float>(0, 2);
        test_point.at<float>(0, 1) = test_point.at<float>(0, 1) / test_point.at<float>(0, 2);
        test_point.at<float>(0, 2) = 1;

        cv::Rect one_point(test_point.at<float>(0, 0), test_point.at<float>(0, 1), 0, 0);

        if (index == 0){
        	bbox = one_point;
        } else {
        	bbox = bbox | one_point;
        }
    }

	return bbox;
}

Mat get_high_freq_mask(Mat img_1){
	return Mat();
}

Mat get_transformation(Mat img_1, Mat img_2){
	int minHessian = 400;
	SiftFeatureDetector detector(minHessian);
	std::vector<KeyPoint> keypoints_1, keypoints_2;

	// Give a mask in the detector. we choose to ignore areas in the image
	// with high fuzziness.
	detector.detect(img_1, keypoints_1);
	detector.detect(img_2, keypoints_2);
	SiftDescriptorExtractor extractor;
	Mat descriptors_1, descriptors_2;

	extractor.compute(img_1, keypoints_1, descriptors_1);
	extractor.compute(img_2, keypoints_2, descriptors_2);

	FlannBasedMatcher matcher;
	std::vector<DMatch> matches;
	matcher.match(descriptors_2, descriptors_1, matches);

	double max_dist = 0; double min_dist = 75;

	//-- Quick calculation of max and min distances between keypoints
	for( int i = 0; i < descriptors_2.rows; i++ )
	{ double dist = matches[i].distance;
	if( dist < min_dist ) min_dist = dist;
	if( dist > max_dist ) max_dist = dist;
	}

	if (false){
		printf("-- Max dist : %f \n", max_dist );
		printf("-- Min dist : %f \n", min_dist );
	}

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

	if (false){
		printf("-- Number of pts for img1 : %ld \n", img_1_forhomography.size() );
		printf("-- Number of pts for img2 : %ld \n", img_2_forhomography.size() );
	}

	// findHomography (from Opencv) returns type of CV_64FC1
	Mat transform = Mat::eye(3, 3, CV_32FC1);
	if (img_1_forhomography.size() >= 4 && img_2_forhomography.size() >= 4){
		transform = findHomography(img_2_forhomography, img_1_forhomography, CV_RANSAC);
		// recast transform to CV_32FC1 type
		transform.convertTo(transform, CV_32FC1);
	}
	return transform;
}

int main(int argc, char* argv[])
{
	// first argument is the number of frames
	int num_frames_to_analyze = atoi(argv[1]);
	int start_index = 0;

	// TODO: get the start index working right
	// if (argc == 3){
	// 	start_index = atoi(argv[2]); 
	// }

	VideoCapture capture("videos/VID_20150418_151108.mp4");
	Mat frame, prev_frame;
	Mat transform = Mat::eye(3, 3, CV_32FC1);
	float decrease_factor = 0.3;
	int blend_type = BLEND_PURE;

	// the panorama that is growing
	Mat trans_img;
	std::vector<Mat> frames;
	std::vector<Mat> transformed_frames;

	int index = 0;
	double frame_width, frame_height, frame_index;

	// the transformations of each image to keep track of
	std::vector<Mat> homographies;

	if(!capture.isOpened()){
		cout << "Error reading video file" << endl;
		return -1;
	} else {
		frame_width = capture.get(CV_CAP_PROP_FRAME_WIDTH);
		frame_height = capture.get(CV_CAP_PROP_FRAME_HEIGHT);
		cout << "frame width = " << frame_width << endl;
		cout << "frame height = " << frame_height << endl;
	}

	while (capture.isOpened() && index < num_frames_to_analyze && index >= start_index){
		frame_index = capture.get(CV_CAP_PROP_POS_FRAMES);
		// cout << "frame index = " << frame_index << endl;
		index++;
		if (frame_index > 0){
			prev_frame = frame;
		}
		capture >> frame;
		if(frame.empty())
            break;
        resize(frame, frame, Size(0, 0), decrease_factor, decrease_factor, INTER_LINEAR);

        // frame *= 1./255;
        // frame.convertTo(frame, CV_32FC3);

        if (false){
        	imshow("original frame", frame);
        	waitKey(20);
        }

        // save the frame as a file
        if (SAVE_FILES){
	        char buffer[200];
	        printf("videos/img_%08d.png", index);
	        sprintf(buffer, "videos/img_%08d.png", index);
	        std::string filename = buffer;
	        imwrite(filename, frame);
        } else {
	        // save the frame in my local memory structure
	        frames.push_back(frame);
        }

        // only compute the transformation when you have two frames
        // otherwise, prev_frame does not exist and there is a segfault
        if (frame_index > 0){
	        transform = get_transformation(frame, prev_frame);
	        if (false){
		        cout << transform << endl;
	        }
        }

        // show the keypoints on a frame
        // for debugging
        if (false){
	        std::vector<KeyPoint> keypoints_1 = get_keypoints(frame);
			drawKeypoints(frame, keypoints_1, frame, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
        }

        homographies.push_back(transform);
    }

    cv::Rect current_pano_size(0, 0, frame_width * decrease_factor, frame_height * decrease_factor);

    for (int i = 0; i < homographies.size(); ++i)
    {
    	transform = homographies[i];

        // calculate bounding box required for next image
        cv::Rect next_bbox = transformed_bbox(frame.cols, frame.rows, transform);
        current_pano_size = current_pano_size | next_bbox;

        // translate the rectangle so that the (x, y) is (0, 0)
        // current_pano_size = current_pano_size - current_pano_size.tl();
        Mat translation = Mat::eye(3, 3, CV_32FC1);
        translation.at<float>(0, 2) = -1*current_pano_size.tl().x;
        translation.at<float>(1, 2) = -1*current_pano_size.tl().y;

        // cout << "old homography: " << homographies[i] << endl;

        homographies[i] = homographies[i] * translation;

        if (false){
        	// cout << "translation matrix: " << translation << endl;
        	// cout << "new homography: " << homographies[i] << endl;
	        printf("next_bbox: %d %d %d %d \n", next_bbox.x, next_bbox.y, next_bbox.width, next_bbox.height);
	        printf("current_pano_size: %d %d %d %d \n", current_pano_size.x, current_pano_size.y, current_pano_size.width, current_pano_size.height);
        }
    }

    // use the final size to create the final pano
    Mat pano(current_pano_size.size(), CV_32FC3, cv::Scalar(0, 0, 0));
    Mat total_pixels_used(current_pano_size.size(), CV_8UC3, cv::Scalar(0, 0, 0));
    // cout << "frame type: " << frame.type() << endl;
    // the number of homographies is the index of the frames minus the first one
    // index--;

    if (true){
		printf("rows: %d, cols: %d \n", pano.rows, pano.cols);
		printf("index: %d, frames length: %lu, homographies length: %lu \n", index, frames.size(), homographies.size());
    	printf("final pano size: %d %d %d %d \n", current_pano_size.x, current_pano_size.y, current_pano_size.width, current_pano_size.height);
    }

    // transform each of the frames and save them
    for (int i = 0; i < index; ++i)
    {
    	frame = frames[i];
    	transform = homographies[i];

        // transform the image according to the transformation
        warpPerspective(frame, trans_img, transform, current_pano_size.size());

        if (false){
	        imshow("transformed", trans_img);
	        waitKey(20);
        }

        trans_img.convertTo(trans_img, CV_32FC3);
        trans_img *= 1./255;

        transformed_frames.push_back(trans_img.clone());
	}

	if (true){
		for (int frame_index = 0; frame_index < index; ++frame_index){
			imshow("transformed_frame", transformed_frames[frame_index]);
			double min, max;
			minMaxLoc(transformed_frames[frame_index], &min, &max);
			cout << "transformed_frames[frame_index]: " << "min = " << min << "; max = " << max << "; type = " << transformed_frames[frame_index].type() << endl;
			waitKey(20);
		}
	}

	// get rid of the vector of original frames, since we only need the 
	// transformed ones now
	frames.clear();
	int thresh_ignore = 50;

	// double min, max;
	// minMaxLoc(pano, &min, &max);
	// cout << "min = " << min << "; max = " << max << endl;

	// minMaxLoc(transformed_frames[0], &min, &max);
	// cout << "min = " << min << "; max = " << max << endl;

	// stitch the panorama with blending
	// for each output pixel, look at each of the potential input pixels
	for (int x = 0; x < pano.rows; ++x) // pano.rows
	{
		for (int y = 0; y < pano.cols; ++y) // pano.cols
		{
			if (false){
				printf("pano coord x: %d, y: %d \n", x, y);
				// printf("final pano size: %d %d %d %d \n", current_pano_size.x, current_pano_size.y, current_pano_size.width, current_pano_size.height);
			}

			if (blend_type == BLEND_MEDIAN){
				std::vector<float> pt_vals;
				for (int frame_index = 0; frame_index < index; ++frame_index){
					pt_vals.push_back(transformed_frames[frame_index].at<float>(x, y));
				}

				// calculate the median
				float median = 0;
				sort(pt_vals.begin(), pt_vals.end());
				if (index % 2 == 0){
					median = (pt_vals[index / 2 - 1] + pt_vals[index / 2]) / 2;
				} else {
					median = pt_vals[index / 2];
				}
				// set the pano value to the median
				pano.at<Vec4f>(x, y) = median;				
			} else if (blend_type == BLEND_PURE){
				int num_avg = 0;
				for (int frame_index = 0; frame_index < index; ++frame_index){
					Vec3f pixel = transformed_frames[frame_index].at<Vec3f>(x, y);
					if (pixel[0] + pixel[1] + pixel[2] > 0.0001){
						num_avg++;
					}
				}

				// for (int frame_index = 0; frame_index < index; ++frame_index){
				// 	// if (norm(transformed_frames[frame_index].at<Vec4b>(x, y)) > thresh_ignore){
				// 		// num_avg++;
				// 	total_pixels_used.at<Vec3i>(x, y) += cv::Vec3i(1, 1, 1);
				// 	// }
				// }

				for (int frame_index = 0; frame_index < index; ++frame_index){
					// if (norm(transformed_frames[frame_index].at<Vec4b>(x, y)) > thresh_ignore){
					// cout << "pano.at<Vec3f>(x, y) = " << pano.at<Vec3f>(x, y) << endl;
					// cout << "transformed_frames[frame_index].at<Vec3f>(x, y):" << transformed_frames[frame_index].at<Vec3f>(x, y) << endl;
					pano.at<Vec3f>(x, y)[0] += transformed_frames[frame_index].at<Vec3f>(x, y)[0];
					pano.at<Vec3f>(x, y)[1] += transformed_frames[frame_index].at<Vec3f>(x, y)[1];
					pano.at<Vec3f>(x, y)[2] += transformed_frames[frame_index].at<Vec3f>(x, y)[2];
					// }
				}

				Vec3f pano_pixel = pano.at<Vec3f>(x, y);
				if (pano_pixel[0] > 0 && pano_pixel[1] > 0 && pano_pixel[2] > 0){
					pano.at<Vec3f>(x, y) /= 1.0 * num_avg;
				}

				// int num_pix = total_pixels_used.at<Vec3f>(x, y)[0];
				// if (num_pix > 0){
				// 	pano.at<Vec3f>(x, y) = pano.at<Vec3f>(x, y) / float(num_pix);
				// }

				// cout << num_avg << " " << pano.at<Vec3f>(x, y) << endl;

				// for (int frame_index = 0; frame_index < index; ++frame_index){
				// 	if (norm(transformed_frames[frame_index].at<Vec3f>(x, y)) > thresh_ignore){
				// 		cout << pano.at<Vec3f>(x, y) << endl;
				// 	}
				// }

				// pano.at<float>(x, y) += transformed_frames[frame_index].at<float>(x, y);
			}
		}
	}

	double min, max;
	minMaxLoc(pano, &min, &max);
	cout << "pano: " << "min = " << min << "; max = " << max << "; type = " << pano.type() << endl;
	// pano.convertTo(pano, CV_8UC3);
	pano /= max;
	minMaxLoc(pano, &min, &max);
	cout << "pano: " << "min = " << min << "; max = " << max << "; type = " << pano.type() << endl;

	minMaxLoc(total_pixels_used, &min, &max);
	cout << "total_pixels_used: " << "min = " << min << "; max = " << max <<"; type = " << total_pixels_used.type() << endl;

	// minMaxLoc(pano, &min, &max);
	// cout << "min = " << min << "; max = " << max << endl;

	// imshow("total_pixels_used", total_pixels_used);
	imshow("finalpano", pano);
	waitKey(0);

	return 0;
}