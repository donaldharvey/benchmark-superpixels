#include "coarsetofine.h"
#include "utils.h"

using namespace cv;

PixelSegmentation run_coarsetofine(cv::Mat& image, int num_superpixels) {
    SPSegmentationParameters params;
    params.superpixelNum = num_superpixels;

    SPSegmentationEngine engine(params, image);
    engine.ProcessImage();

    Mat output = engine.GetSegmentation();
    Mat_<int32_t> labels;
    output.convertTo(labels, CV_32S);
    labels += 1;
    
    Mat_<uchar> contours = generate_boundary_mat(labels);
    return PixelSegmentation(labels, contours);
}
