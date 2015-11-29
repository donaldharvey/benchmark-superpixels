#include "segmentation.h"
#include <cstdint>
#include <fstream>
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>

// Segmentation::boundary_recall() {
//     if
// }

PixelSegmentation::PixelSegmentation(int32_t* data, uchar* b_data, int32_t width, int32_t height) : segmentation_data(height, width, CV_32S, data), boundary_data(height, width, CV_8U, b_data) {
    this->height = height;
    this->width = width;
    PixelSegmentation::initialise_segments();
}

PixelSegmentation::PixelSegmentation(cv::Mat& data, cv::Mat& boundary_data) : segmentation_data(data), boundary_data(boundary_data) {
    this->height = data.rows;
    this->width = data.cols;
    PixelSegmentation::initialise_segments();   
}

PixelSegmentation::PixelSegmentation() {}

PixelSegment::PixelSegment(PixelSegmentation& seg) : segmentation(seg) {}

void PixelSegmentation::initialise_segments() {
    double min, max;
    cv::minMaxIdx(segmentation_data, &min, &max);
    int32_t number_segments = int32_t(max);
    this->segments.reserve(number_segments);

    for(int l = 1; l <= number_segments; ++l) {
        PixelSegment p(*this);
        p.label = l;
        p.x1 = p.x2 = p.y1 = p.y2 = -1;
        this->segments.push_back(p);
    }

    for(int i = 0; i < segmentation_data.rows; ++i) {
        for(int j = 0; j < segmentation_data.cols; ++j) {
            int32_t label = segmentation_data.at<int32_t>(i,j);
            if(label==0) { std::cout << "ZERO @ " << i << "," << j << std::endl; continue; }
            PixelSegment& s = segments[label-1];
            s.xs.push_back(j);
            s.ys.push_back(i);
            if(s.x1 < 0 or s.x1 > j) { s.x1 = j; }
            if(s.x2 < 0 or s.x2 < j) { s.x2 = j; }
            if(s.y1 < 0 or s.y1 > i) { s.y1 = i; }
            if(s.y2 < 0 or s.y2 < i) { s.y2 = i; }
        }
    }

    // for(auto i = segments.begin(); i != segments.end(); ++i) {
    //     std::cout << i->x1 << " " << i->x2 << " " << i->label << std::endl;
    // }
}

PixelSegmentation& PixelSegmentation::load_from_file(ifstream& labels_file) {
    // format: two int32s specifying width and height, followed by w*h int32s.
    int32_t* dims = new int32_t[2];
    labels_file.read((char*)dims, sizeof(int32_t)*2);
    int32_t width = dims[0];
    int32_t height = dims[1];
    int32_t* data = new int32_t[width*height];
    uchar* bdata = new uchar[width*height];
    labels_file.read((char*)data, width*height*sizeof(int32_t));
    labels_file.read((char*)bdata, width*height*sizeof(uchar));
    PixelSegmentation* p = new PixelSegmentation(data, bdata, width, height);
    return *p;
}

cv::Mat PixelSegmentation::get_boundary_pixels() const {
//     cv::Mat image = cv::Mat(this->height, this->width, CV_8U, 0.0);
//     for(int i = 0; i < this->segmentation_data.rows; ++i) {
//         for(int j = 0; j < this->segmentation_data.cols; ++j) {
//             int32_t label = this->segmentation_data.at<int32_t>(i,j);
//             int dy8[8] = {-1, 0, 1, 0, -1, -1, 1, 1};
//             int dx8[8] = {0, 1, 0, -1, -1, 1, -1, 1};
//             for(int di=0; di<8;++di) {
//                 int x = j + dx8[di], y = i + dy8[di];
//                 if(x >= 0 and x < this->width and y >= 0 and y < this->height) {
//                     if(this->segmentation_data.at<int32_t>(y,x) != label) {
//                         // it's a boundary pixel
//                         image.at<uchar>(i, j) = (uchar)1;
//                         break;
//                     }
//                 }
//             }
//         }
//     }
//     return image;
    return this->boundary_data;
}

double PixelSegmentation::boundary_recall(PixelSegmentation& ground_truth, int epsilon) {
    int true_pos = 0, false_neg = 0;
    cv::Mat ground_truth_boundaries = ground_truth.get_boundary_pixels();
    cv::Mat our_boundaries = this->get_boundary_pixels();
    for(int p_y = 0; p_y < ground_truth_boundaries.rows; ++p_y) {
        const uchar* Mi = ground_truth_boundaries.ptr(p_y);
        for(int p_x = 0; p_x < ground_truth_boundaries.cols; ++p_x) {
            if (Mi[p_x]) {
                // look in area about (p_y, p_x)...
                cv::Range y_range = cv::Range(max(p_y-epsilon,0), min(p_y+epsilon+1, this->height));
                cv::Range x_range = cv::Range(max(p_x-epsilon,0), min(p_x+epsilon+1, this->width));
                cv::Mat test_area = our_boundaries(y_range, x_range);
                auto res = cv::countNonZero(test_area);
                if(res) {
                    //std::cout << "Increment." << std::endl;
                    true_pos += 1;
                }
                else {
                    //std::cout << "fincrement." << std::endl;
                    false_neg += 1;
                }
            }
        }
    }
    return (double)true_pos / (true_pos + false_neg);
}

bool PixelSegment::bbox_intersect(PixelSegment& other) {
    // std::cout << "ax1 < bx2, " << (a.x1 <= b.x2) << std::endl;
    // std::cout << "ax2 > bx1, " << (a.x2 >= b.x1) << std::endl;
    // std::cout << "ay1 < by2, " << (a.y1 <= b.y2) << std::endl;
    // std::cout << "ay2 > by1, " << (a.y2 >= b.y1) << "\n\n";
    // std::cout << "OVERALL" << (a.x1 <= b.x2 and a.x2 >= b.x1 and a.y1 <= b.y2 and a.y2 >= b.y1) << std::endl;
    return (this->x1 <= other.x2 and this->x2 >= other.x1 and this->y1 <= other.y2 and this->y2 >= other.y1);
}

intersection_result PixelSegment::intersection(PixelSegment& other) {
    int area_in = 0, area_out = 0;
    for(int y = this->y1; y<this->y2; ++y) {
        for(int x = this->x1; x<this->x2; ++x) {
            if(this->segmentation.segmentation_data.at<int32_t>(y,x) == this->label) {
                if(other.segmentation.segmentation_data.at<int32_t>(y,x) == other.label) {
                    ++area_in;
                }
                else {
                    ++area_out;
                }
            }
        }
    }
    return (intersection_result){.area_out=area_out, .area_in=area_in};

}

double PixelSegmentation::undersegmentation_error(PixelSegmentation& ground_truth) {
    int total = 0;
    for(auto i = ground_truth.segments.begin(); i != ground_truth.segments.end(); i++) {
        auto gt_seg = *i;
        // find superpixels whose boundary boxes intersect...
        for(auto j = this->segments.begin(); j != this->segments.end(); j++) {
            auto this_seg = *j;
            if (!this_seg.bbox_intersect(gt_seg)) {
                continue; 
            }
            intersection_result result = this_seg.intersection(gt_seg);
            if(result.area_in) {
                total += min(result.area_in, result.area_out);
            }
        }
    }
    return (double)total / (this->width * this->height);
}

double PixelSegment::perimeter() {
    int total = 0;
    auto len = this->xs.size();
    for(int i = 0; i < len; ++i) {
        // is it a boundary point?
        cv::Point point(this->xs[i], this->ys[i]);
        cv::Point diffs[4] = {cv::Point(1,0),cv::Point(-1,0),cv::Point(0,1),cv::Point(0,-1)};
        for(int j=0; j<4; ++j) {
            cv::Point p = point + diffs[j];
            if(p.x < 0 or p.y < 0 or p.x >= this->segmentation.width or p.y >= this->segmentation.height) {
                continue;
            }
            if(this->label != this->segmentation.segmentation_data.at<int32_t>(p)) {
                ++total;
            }
        }
    }
    return (double)total;
}

double PixelSegment::area() { return this->xs.size(); }

double PixelSegmentation::achievable_segmentation_accuracy(PixelSegmentation& ground_truth) {
    int total = 0;
    for(auto i = this->segments.begin(); i != this->segments.end(); i++) {
        // find intersecting gts and add the largest interesection
        PixelSegment this_seg = *i;

        int max = 0;
        for(auto j = ground_truth.segments.begin(); j != ground_truth.segments.end(); j++) {
            PixelSegment gt_seg = *j;
            if (!this_seg.bbox_intersect(gt_seg)) { continue; }
            intersection_result result = this_seg.intersection(gt_seg);
            if (result.area_in > max) {
                max = result.area_in;
            }
        }
        total += max;
    }
    return (double)total / (this->width * this->height);
}

double PixelSegmentation::compactness() {
    double CO = 0;
    for(auto i = this->segments.begin(); i != this->segments.end(); i++) {
        CO += pow(i->area() / i->perimeter(), 2);
    }
    CO *= 4*M_PI / (this->width * this->height);
    return CO;
}

void PixelSegmentation::compute_mean(cv::Mat& image, cv::Mat& output) {
    vector<cv::Vec3d> means(this->number_segments(), 0.0);
    for(int p_y = 0; p_y < image.rows; ++p_y) {
        for(int p_x = 0; p_x < image.cols; ++p_x) {
            means[this->segmentation_data.at<int32_t>(p_y, p_x) - 1] += image.at<cv::Vec3b>(p_y, p_x);
        }
    }
    for(int i=0; i<means.size(); ++i) {
        means[i] /= this->segments[i].area();
    }
    for(int p_y = 0; p_y < image.rows; ++p_y) {
        for(int p_x = 0; p_x < image.cols; ++p_x) {
            output.at<cv::Vec3b>(p_y, p_x) = means[this->segmentation_data.at<int32_t>(p_y, p_x) - 1];
        }
    }
}

double PixelSegmentation::reconstruction_error(const cv::Mat& image) {
    cv::Mat greyscale_image;
    cv::cvtColor(image, greyscale_image, CV_BGR2GRAY);

    std::vector<int> colours;

    // compute the average colour for each face.
    for(int label=1; label <= this->number_segments(); ++label) {
        int pixel_count = 0;
        double sum = 0;
        for(int p_y = 0; p_y < greyscale_image.rows; ++p_y) {
            const uchar* Mi = greyscale_image.ptr(p_y);
            for(int p_x = 0; p_x < greyscale_image.cols; ++p_x) {
                if(this->segmentation_data.at<int32_t>(p_y, p_x) == label) {
                    ++pixel_count;
                    sum += Mi[p_x];
                }
            }
        } 
        colours.push_back((sum/255)/pixel_count);
    }

    double total_error = 0;


    for(int p_y = 0; p_y < greyscale_image.rows; ++p_y) {
        const uchar* Mi = greyscale_image.ptr(p_y);
        for(int p_x = 0; p_x < greyscale_image.cols; ++p_x) { 
            // pixel error = intensity(p) - ∑(face colours weighted by area of intersection)
            double pixel_error = (double)Mi[p_x]/255;
            // for a pixel-based segmentation, rather than loop over faces, just get the label at this index.
            int label = this->segmentation_data.at<int>(p_y, p_x);
            pixel_error -= colours[label-1];

            //std::cout << pixel_error << std::endl;
            pixel_error = std::pow(pixel_error, 2);
            total_error += pixel_error;
        }
    }

    return total_error / image.total();
}

unsigned long PixelSegmentation::number_segments() {
    return this->segments.size();
}



