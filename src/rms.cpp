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
    double re = seg.reconstruction_error(image);
    double nre = seg.normalised_reconstruction_error(re);
    cout.precision(10);
    cout << re << " " << nre << endl;
}
