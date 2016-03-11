#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv; 

Mat_<uchar> generate_boundary_mat(const Mat_<int32_t>& labels);
const inline int get_label_at(const cv::Mat_<int32_t>& seg, int i, int j);
Mat_<uchar> thin_boundary_matrix(Mat_<uchar>& mat);