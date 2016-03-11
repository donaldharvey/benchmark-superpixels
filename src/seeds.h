//
//  run-seeds.h
//  superpixel-benchmarks
//
//  Created by Donald S. F. Harvey on 27/11/2015.
//
//

#ifndef run_seeds_h
#define run_seeds_h

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ximgproc/seeds.hpp>
#include <fstream>
#include <iostream>
#include "segmentation.h"

using namespace cv;

Mat_<int32_t> run_seeds(cv::Mat& image, int num_superpixels, int num_levels=4);

#endif /* run_seeds_h */
