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

PixelSegmentation run_seeds(cv::Mat& image, int num_superpixels);

#endif /* run_seeds_h */
