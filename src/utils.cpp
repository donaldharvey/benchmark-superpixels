#include "utils.h"

using namespace cv; 

Mat generate_boundaries(Mat& labels) {
    // from slic implementation.

    Mat boundaries(labels.rows,labels.cols,CV_8U, 0.0);
    const int dx8[8] = {-1, -1,  0,  1, 1, 1, 0, -1};
    const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1,  1};

    /* Go through all the pixels. */
    for (int i = 0; i < boundaries.rows; i++) {
        for (int j = 0; j < boundaries.cols; j++) {
            int nr_p = 0;
            
            /* Compare the pixel to its 8 neighbours. */
            for (int k = 0; k < 8; k++) {
                int x = j + dx8[k], y = i + dy8[k];
                
                if (x >= 0 && x < boundaries.cols && y >= 0 && y < boundaries.rows) {
                    if (boundaries.at<uchar>(y, x) == 0 && labels.at<int>(i, j) != labels.at<int>(y,x)) {
                        nr_p += 1;
                    }
                }
            }
            
            /* Add the pixel to the contour list if desired. */
            if (nr_p >= 2) {
                boundaries.at<uchar>(i,j) = 1;
            }
        }
    }
    return boundaries;
}
