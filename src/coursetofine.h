//
//  run-coursetofine.h
//  superpixel-benchmarks
//
//  Created by Donald S. F. Harvey on 27/11/2015.
//
//

#ifndef run_coursetofine_h
#define run_coursetofine_h

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <fstream>
#include <iostream>
#include "segmentation.h"
#include "contrib/coursetofine/segengine.h"

using namespace cv;

PixelSegmentation run_coursetofine(cv::Mat& image, int num_superpixels);

#endif /* run_coursetofine_h */
