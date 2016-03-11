//
//  run-seeds.cpp
//  superpixel-benchmarks
//
//  Created by Donald S. F. Harvey on 29/11/2015.
//
//

#include "seeds.h"
#include <boost/filesystem.hpp>

using namespace cv;

int main(int argc, char *argv[]) {
    using namespace std;
    using namespace cv::ximgproc;

    // Load the image.
    boost::filesystem::path p(argv[1]);
    string root_name = p.stem().string();
    
    Mat image;
    image = imread(argv[1], CV_LOAD_IMAGE_COLOR);

    auto res = run_seeds(image, atoi(argv[2]), atoi(argv[3]));


    imwrite(root_name + ".png", res);
    return 0;
}