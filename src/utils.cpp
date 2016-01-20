#include "utils.h"
#include <array>
#include <iostream>

using namespace cv; 
using namespace std;


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

const inline bool __satisfies_1_and_2(array<uchar, 8> &nhood) {
    // cond 1
    int x_h = 0;

    // cond 2
    int n_1 = 0;
    int n_2 = 0;

    for(int i=0; i<4; ++i) {
        x_h += (!nhood[2*i - 2] && (nhood[2*i-1] || nhood[2*i]));
        n_1 += nhood[2*i - 2] || nhood[2*i - 1];
        n_2 += nhood[2*i - 1] || nhood[2*i];
    }
    int m = std::min(n_1, n_2);
    return (x_h == 1 && (m == 2 || m == 3));
}

array<uchar, 8> &fill_nhood(array<uchar,8> &nhood, int y, int x, Mat_<uchar> &mat, Rect &rect) {
    int dx8[8] = {1, 1, 0, -1, -1, -1, 0, 1};
    int dy8[8] = {0, -1, -1, -1, 0, 1, 1, 1};
    for(int i=0; i<8; ++i) {
        Point idx = Point(x+dx8[i], y+dy8[i]);
        if (rect.contains(idx)) {
            nhood[i] = mat(idx.y, idx.x);
        }
        else {
            nhood[i] = 0;
        }
    };
    return nhood;
}

Mat_<uchar> thin_boundary_matrix(Mat_<uchar> mat) {
    Rect rect(Point(), mat.size());
    bool has_changed = false;
    int number_it = 0;
    do {
        number_it += 1;
        has_changed = false;
        array<uchar, 8> nhood;
        for (int i = 0; i < mat.rows; i++) {
            for (int j = 0; j < mat.cols; j++) {
                if (!mat(i,j))
                    continue;
                
                fill_nhood(nhood, i, j, mat, rect);

                if (__satisfies_1_and_2(nhood)) {
                    if ((nhood[1] || nhood[2] || !nhood[7]) && nhood[0]) {
                        mat(i,j) = 0;
                        has_changed = true;
                    }
                }
            }
        }
        for (int i = 0; i < mat.rows; i++) {
            for (int j = 0; j < mat.cols; j++) {
                if (!mat(i,j))
                    continue;
                
                fill_nhood(nhood, i, j, mat, rect);

                if (__satisfies_1_and_2(nhood)) {
                    if ((nhood[5] || nhood[6] || !nhood[3]) && nhood[4]) {
                        mat(i,j) = 0;
                        has_changed = true;
                    }
                }
            }
        }
    } while (has_changed);
    cout << number_it << " iterations." << endl;
    return mat;
}