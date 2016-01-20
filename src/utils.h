#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv; 

Mat generate_boundaries(Mat& labels);
Mat_<uchar> thin_boundary_matrix(Mat_<uchar> mat);