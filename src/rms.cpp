#include "segmentation.h"
#include <iostream>
#include <fstream>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include <string>
#include <map>

// namespace po = boost::program_options;

using namespace std;

int main(int ac, char* argv[]) {
    string loc = string(argv[2]);
    PixelSegmentation seg = PixelSegmentation::load_from_png(loc);
    cv::Mat image = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);
    cv::Mat m;
    cv::Mat m2;
    seg.compute_mean(image, m2);
    seg.draw_boundaries(image, m);
    cv::imwrite("out-means.png", m2);
    cv::imwrite("out-bounds.png", m);
}
