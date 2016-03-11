//
//  donald-slic.h
//  superpixel-benchmarks
//
//  Created by Donald S. F. Harvey on 24/11/2015.
//
//

#ifndef donald_slic_h
#define donald_slic_h
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "segmentation.h"
#include <fstream>
#include <iostream>

using namespace cv;
#define NR_ITERATIONS 10

Mat_<int32_t> run_slic(Mat& image, int target_superpixel_number, int m);


#endif /* donald_slic_h */
