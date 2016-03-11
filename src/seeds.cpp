#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ximgproc/seeds.hpp>
#include <fstream>
#include <iostream>
#include "segmentation.h"

using namespace cv;

Mat_<int32_t> run_seeds(cv::Mat& image, int num_superpixels, int num_levels) {
    cv::Mat lab_image;
    cv::cvtColor(image, lab_image, CV_BGR2Lab);
    
    int width = image.cols, height = image.rows;
    
    // compute the levels parameter.. see https://github.com/davidstutz/seeds-revised/blob/master/lib/SeedsRevised.cpp#L78
    // int minimum_block_widths[5] = {2, 3, 4, 5, 6};
    // int minimum_block_heights[5] = {2, 3, 4, 5, 6};
    // int max_levels = 12;
    
    // int min_difference = -1;
    // int min_levels = 0;
    // int min_block_width = 0;
    // int min_block_height = 0;
    
    // for (int w = 0; w < 5; ++w) {
    //     for (int h = 0; h < 5; ++h) {
    //         if (abs(minimum_block_widths[w] - minimum_block_heights[h]) > 1) {
    //             continue;
    //         }
            
    //         for (int l = 2; l < max_levels + 1; ++l) {
    //             int superpixels = std::floor(width/(minimum_block_widths[w]*pow(2, l - 1))) * std::floor(height/(minimum_block_heights[h]*pow(2, l - 1)));
    //             int difference = abs(num_superpixels - superpixels);
    //             if (difference < min_difference || min_difference < 0) {
    //                 min_difference = difference;
    //                 min_levels = l;
    //                 min_block_width = minimum_block_widths[w];
    //                 min_block_height = minimum_block_heights[h];
    //             }
    //         }
    //     }
    // }
    
    auto seeds = cv::ximgproc::createSuperpixelSEEDS(width, height, 3, num_superpixels, num_levels, 2, 5, false);
    
    //    auto seeds = createSuperpixelSEEDS(w, h, 3, atoi(argv[2]));
    seeds->iterate(lab_image, 4);
    
    Mat_<int32_t> seedsLabels;
    seeds->getLabels(seedsLabels);
    seedsLabels += 1;
    
    cout << seeds->getNumberOfSuperpixels() << endl;
    
    return seedsLabels;
}
