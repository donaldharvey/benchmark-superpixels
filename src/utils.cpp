#include "utils.h"
#include <array>
#include <iostream>
#include <opencv2/highgui/highgui.hpp> // deleteme

using namespace cv; 
using namespace std;


const inline int get_label_at(const cv::Mat_<int32_t>& seg, int i, int j) {
    if (i < 0 || i >= seg.rows || j < 0 || j >= seg.cols) {
        return -1;
    }
    else {
        return seg(i,j);
    }
}

Mat_<uchar> generate_boundary_mat(const Mat_<int32_t>& seg) {
    // initiate mat 2x bigger
    cv::Mat_<uchar> scaled = Mat_<uchar>(2*seg.rows+1, 2*seg.cols+1, 0.0);

    cv::Mat_<uchar> image = Mat_<uchar>(seg.rows, seg.cols, 0.0);

    for(int i = 0; i < seg.rows; ++i) {
        for(int j = 0; j < seg.cols; ++j) {
            
            int p = get_label_at(seg, i, j);
            int s = get_label_at(seg, i+1, j);
            int s2 = get_label_at(seg, i+2, j);
            int se = get_label_at(seg, i, j);
            int e = get_label_at(seg, i, j+1);
            int e2 = get_label_at(seg, i, j+2);

            // hij = pij != pij+1
            // hi+1j = pi+1j != pi+1j+1
            
            uchar south_diff = s != -1 && p != s;
            uchar east_diff = e != -1 && p != e;
            uchar next_east_diff = s != se && s != -1 && e != -1;
            uchar next_south_diff = e != se && s != -1 && e != -1;
            

            scaled(2+2*i, 1+2*j) = south_diff; // (odd, even)
            scaled(1+2*i, 2+2*j) = east_diff;  // (even, odd)
            scaled(2+2*i, 2+2*j) = south_diff or east_diff or next_south_diff or next_east_diff;
        }
    }

    for(int i=0; i < scaled.rows; ++i) {
        scaled(i, 0) = scaled(i, 1);
        scaled(i, scaled.cols-1) = scaled(i, scaled.cols-2);
    }

    for(int j=0; j < scaled.cols; ++j) {
        scaled(0, j) = scaled(0, j+1);
        scaled(scaled.rows-1, j) = scaled(scaled.rows-2, j);
    }

    for(int i = 0; i < seg.rows; ++i) {
        for(int j = 0; j < seg.cols; ++j) {
            image(i,j) = scaled(2+2*i, 2+2*j);
        }
    }
    
    imwrite("scaled.png", scaled*255);

    return image;
}


const inline bool __satisfies_1_and_2(array<uchar, 8> &nhood) {
    // cond 1
    int x_h = 0;

    // cond 2
    int n_1 = 0;
    int n_2 = 0;

    for(int i=0; i<4; ++i) {
        x_h += bool(!nhood[2*i] && (nhood[2*i + 1] || nhood[(2*i + 2)%8]));
        if (x_h >= 2) {
            return false;
        }
        n_1 += bool(nhood[2*i] || nhood[2*i + 1]);
        n_2 += bool(nhood[2*i + 1] || nhood[(2*i + 2)%8]);
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

Mat_<uchar> thin_boundary_matrix(Mat_<uchar>& mat) {
    Rect rect(Point(), mat.size());
    bool has_changed = false;
    int number_it = 0;
    do {
        Mat_<uchar> newmat(mat.rows, mat.cols, 0.0);
        int number_satisfactions = 0;
        number_it += 1;
        has_changed = false;
        array<uchar, 8> nhood;
        for (int i = 0; i < mat.rows; i++) {
            for (int j = 0; j < mat.cols; j++) {
                if (!mat(i,j))
                    continue;
                newmat(i,j) = mat(i,j);
                
                fill_nhood(nhood, i, j, mat, rect);

                if (__satisfies_1_and_2(nhood)) {
                    if (bool((nhood[1] || nhood[2] || !nhood[7]) && nhood[0]) == 0) {
                        newmat(i,j) = 0;
                        number_satisfactions += 1;
                        has_changed = true;
                    }
                }
            }
        }
        mat = newmat.clone();

        for (int i = 0; i < mat.rows; i++) {
            for (int j = 0; j < mat.cols; j++) {
                if (!mat(i,j))
                    continue;
                newmat(i,j) = mat(i,j);
                
                fill_nhood(nhood, i, j, mat, rect);

                if (__satisfies_1_and_2(nhood)) {
                    if (bool((nhood[5] || nhood[6] || !nhood[3]) && nhood[4]) == 0) {
                        newmat(i,j) = 0;
                        number_satisfactions += 1;
                        has_changed = true;
                    }
                }
            }
        }
        mat = newmat.clone();
    } while (has_changed);
    return mat;
}