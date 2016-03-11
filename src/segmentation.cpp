#include "segmentation.h"
#include <cstdint>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <iterator>
#include <unordered_set>
#include "utils.h"
#include <opencv2/imgproc/imgproc.hpp>

#include <opencv2/highgui/highgui.hpp> // deleteme

// Segmentation::boundary_recall() {
//     if
// }

Mat3d convert_rgb_to_lab(const Mat& img)
{
    const int RGB2LABCONVERTER_XYZ_TABLE_SIZE = 1024;
    // CIE standard parameters
    const double epsilon = 0.008856;
    const double kappa = 903.3;
    // Reference white
    const double referenceWhite[3] = { 0.950456, 1.0, 1.088754 };
    /// Maximum values
    const double maxXYZValues[3] = { 0.95047, 1.0, 1.08883 };
    
    vector<float> sRGBGammaCorrections(256);
    for (int pixelValue = 0; pixelValue < 256; ++pixelValue) {
        double normalizedValue = pixelValue / 255.0;
        double transformedValue = (normalizedValue <= 0.04045) ? normalizedValue / 12.92 : pow((normalizedValue + 0.055) / 1.055, 2.4);
        
        sRGBGammaCorrections[pixelValue] = transformedValue;
    }
    
    int tableSize = RGB2LABCONVERTER_XYZ_TABLE_SIZE;
    vector<double> xyzTableIndexCoefficients(3);
    xyzTableIndexCoefficients[0] = (tableSize - 1) / maxXYZValues[0];
    xyzTableIndexCoefficients[1] = (tableSize - 1) / maxXYZValues[1];
    xyzTableIndexCoefficients[2] = (tableSize - 1) / maxXYZValues[2];
    
    vector<vector<float> > fXYZConversions(3);
    for (int xyzIndex = 0; xyzIndex < 3; ++xyzIndex) {
        fXYZConversions[xyzIndex].resize(tableSize);
        double stepValue = maxXYZValues[xyzIndex] / tableSize;
        for (int tableIndex = 0; tableIndex < tableSize; ++tableIndex) {
            double originalValue = stepValue*tableIndex;
            double normalizedValue = originalValue / referenceWhite[xyzIndex];
            double transformedValue = (normalizedValue > epsilon) ? pow(normalizedValue, 1.0 / 3.0) : (kappa*normalizedValue + 16.0) / 116.0;
            
            fXYZConversions[xyzIndex][tableIndex] = transformedValue;
        }
    }
    
    Mat3d result = Mat3d(img.rows, img.cols);
    
    for (int y = 0; y < img.rows; ++y) {
        for (int x = 0; x < img.cols; ++x) {
            const Vec3b& rgbColor = img.at<Vec3b>(y, x);
            Vec3d& labColor = result(y, x);
            
            float correctedR = sRGBGammaCorrections[rgbColor[2]];
            float correctedG = sRGBGammaCorrections[rgbColor[1]];
            float correctedB = sRGBGammaCorrections[rgbColor[0]];
            float xyzColor[3];
            
            xyzColor[0] = correctedR*0.4124564f + correctedG*0.3575761f + correctedB*0.1804375f;
            xyzColor[1] = correctedR*0.2126729f + correctedG*0.7151522f + correctedB*0.0721750f;
            xyzColor[2] = correctedR*0.0193339f + correctedG*0.1191920f + correctedB*0.9503041f;
            
            int tableIndexX = static_cast<int>(xyzColor[0] * xyzTableIndexCoefficients[0] + 0.5);
            int tableIndexY = static_cast<int>(xyzColor[1] * xyzTableIndexCoefficients[1] + 0.5);
            int tableIndexZ = static_cast<int>(xyzColor[2] * xyzTableIndexCoefficients[2] + 0.5);
            
            float fX = fXYZConversions[0][tableIndexX];
            float fY = fXYZConversions[1][tableIndexY];
            float fZ = fXYZConversions[2][tableIndexZ];
            
            labColor[0] = 116.0*fY - 16.0;
            labColor[1] = 500.0*(fX - fY);
            labColor[2] = 200.0*(fY - fZ);
        }
    }
    return result;
}

PixelSegmentation::PixelSegmentation(int32_t* data, uchar* b_data, int32_t width, int32_t height) : segmentation_data(height, width, data), boundary_data(height, width, b_data) {
    this->height = height;
    this->width = width;
    PixelSegmentation::initialise_segments();
}

PixelSegmentation::PixelSegmentation(cv::Mat_<int32_t>& data, cv::Mat_<uchar>& boundary_data) : segmentation_data(data), boundary_data(boundary_data) {
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
            s.points.push_back(cv::Point(j,i));
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

PixelSegmentation PixelSegmentation::load_from_png(string& path) {
    // format: two int32s specifying width and height, followed by w*h int32s.
    cv::Mat input = cv::imread(path, CV_LOAD_IMAGE_GRAYSCALE);
    cv::Mat_<int32_t> seg_mat = input + 1;
    cv::Mat_<uchar> b = generate_boundary_mat(seg_mat);
    b = thin_boundary_matrix(b);
    return PixelSegmentation(seg_mat, b);
}

PixelSegmentation PixelSegmentation::load_from_file(ifstream& labels_file) {
    // format: two int32s specifying width and height, followed by w*h int32s.
    int32_t* dims = new int32_t[2];
    labels_file.read((char*)dims, sizeof(int32_t)*2);
    int32_t width = dims[0];
    int32_t height = dims[1];
    int32_t* data = new int32_t[width*height];
    uchar* bdata = new uchar[width*height];
    labels_file.read((char*)data, width*height*sizeof(int32_t));
    labels_file.read((char*)bdata, width*height*sizeof(uchar));
    cv::Mat_<int32_t> seg_mat = cv::Mat_<int32_t>(height, width, data).clone();
    cv::Mat_<uchar> boundary_mat = cv::Mat_<uchar>(height, width, bdata).clone();
    delete data;
    delete bdata;
    delete dims;
    return PixelSegmentation(seg_mat, boundary_mat);
}

//void PixelSegmentation::output_to_file(ofstream& out_file) {
//    uchar* contoursOut = NULL;
//    int32_t* regionsOut = NULL;
//    
//    contoursOut = new uchar[this->width * this->height];
//    regionsOut = new int32_t[this->width * this->height];
//
//    for (int i = 0; i < this->height; i++) {
//        for (int j = 0; j < this->width; j++) {
//            contoursOut[i*this->width + j] = this->boundary_data.at<uchar>(i, j);
//            regionsOut[i*this->width + j] = int32_t(this->segmentation_data.at<int>(i, j));
//        }
//    }
//
//    out_file.write((char*)&(this->width), sizeof(int32_t));
//    out_file.write((char*)&(this->height), sizeof(int32_t));
//    
//    for (int i=0; i < (this->width) * (this->height); i++) {
//        out_file.write((char*)&(regionsOut[i]), sizeof(int32_t));
//    }
//    
//    for (int i=0; i < (this->width) * (this->height); ++i) {
//        out_file.write((char*)&(contoursOut[i]), sizeof(uchar));
//    }
//    out_file.close();
//}

const inline int32_t PixelSegmentation::label_at(int i, int j) {
    return get_label_at(this->segmentation_data, i, j);
}

cv::Mat_<uchar> PixelSegmentation::get_boundary_pixels() const {
    return this->boundary_data;
    // cv::Mat_<uchar> image = generate_boundary_mat(this->segmentation_data);
    // cv::Mat outPreThin;
    // cv::Mat outPostThin;
    // cv::cvtColor(image*255, outPreThin, CV_GRAY2BGR);
    // // imwrite("outPreThin.png", outPreThin);
    // // now thin.
    // image = thin_boundary_matrix(image);
    // cv::cvtColor(image*255, outPostThin, CV_GRAY2BGR);
    // // imwrite("outPostThin.png", outPostThin);
    // return image;
}



//    // do the initial step
//    cv::Mat_<uchar> image = cv::Mat_<uchar>(this->height, this->width, 0.0);
//    for(int i = 0; i < this->segmentation_data.rows; ++i) {
//        for(int j = 0; j < this->segmentation_data.cols; ++j) {
//            int32_t px = this->segmentation_data(i,j);
//            // check if last row
//            if (i + 1 == this->segmentation_data.rows) {
//                if (j + 1 == this->segmentation_data.cols) {
//                    image(i,j) = 0;
//                }
//                else {
//                    image(i,j) = (px != this->segmentation_data(i, j+1));
//                }
//            }
//            else {
//                if (j + 1 == this->segmentation_data.cols) {
//                    image(i,j) = (px != this->segmentation_data(i+1, j));
//                }
//                else {
//                    uint32_t pixel_value = (px != this->segmentation_data(i+1,j) || 
//                                            px != this->segmentation_data(i,j+1) || 
//                                            px != this->segmentation_data(i+1,j+1));
//                    image(i,j) = pixel_value;
//                }
//            }
//        }
//    }
//}

double PixelSegmentation::boundary_recall(PixelSegmentation& ground_truth, int epsilon) {
    int true_pos = 0, false_neg = 0;
    cv::Mat_<uchar> ground_truth_boundaries = ground_truth.get_boundary_pixels();
    cv::Mat_<uchar> our_boundaries = this->get_boundary_pixels();
    for(int p_y = 0; p_y < ground_truth_boundaries.rows; ++p_y) {
        const uchar* Mi = ground_truth_boundaries.ptr(p_y);
        for(int p_x = 0; p_x < ground_truth_boundaries.cols; ++p_x) {
            if (Mi[p_x]) {
                // look in area about (p_y, p_x)...
                cv::Range y_range = cv::Range(max(p_y-epsilon,0), min(p_y+epsilon+1, this->height));
                cv::Range x_range = cv::Range(max(p_x-epsilon,0), min(p_x+epsilon+1, this->width));
                auto test_area = our_boundaries(y_range, x_range);
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

bool comparePoints(const cv::Point & a, const cv::Point & b) {
    if (a.y == b.y) {
        return a.x < b.x;
    } else {
        return a.y < b.y;
    }
}

template <class T>
inline void hash_combine(std::size_t& seed, const T& v)
{
    std::hash<T> hasher;
    seed ^= hasher(v) + 0x9e3779b9 + (seed<<6) + (seed>>2);
}

namespace std {
    template <> struct hash<cv::Point> {
        size_t operator()(const cv::Point & p) const {
            size_t seed = 0;
            hash_combine(seed, p.x);
            hash_combine(seed, p.y);
            return seed;
        }
    };
}

intersection_result PixelSegment::intersection_and_diff(PixelSegment& other) {

    int in = 0, out = 0;

    // for(auto &i: other.points) {
    //     if (points.find(i) != points.end())
    //         in++;
    //     else
    //         out++;
    // }
    in = std::count_if(other.points.cbegin(), other.points.cend(), [&](Point element) {
        return std::binary_search(points.cbegin(), points.cend(), element, comparePoints);
    });
    out = area() - in;
    return {.area_out=out, .area_in=in};
}

double PixelSegmentation::symmetric_undersegmentation_error(PixelSegmentation& ground_truth) {
    double total = 0;
    for(auto &gt_seg: ground_truth.segments) {
        // find superpixels whose boundary boxes intersect...
        for(auto &this_seg: segments) {
            if (!gt_seg.bbox_intersect(this_seg)) {
                continue; 
            }
            
            intersection_result result = this_seg.intersection_and_diff(gt_seg);
            if(result.area_in) {
                total += min(result.area_in, result.area_out);
            }
        }
    }
    return total / (this->width * this->height);
}

double PixelSegmentation::corrected_undersegmentation_error(PixelSegmentation& ground_truth) {
    double total = 0;
    for (auto &seg: segments) {
        int best_gt_overlap = 0;
        for (auto &gt_seg: ground_truth.segments) {
            if (!gt_seg.bbox_intersect(seg)) {
                continue;
            }
            auto result = seg.intersection_and_diff(gt_seg);
            if(result.area_in > best_gt_overlap) {
                best_gt_overlap = result.area_in;
            }
        }
        total += abs(seg.area() - best_gt_overlap);
    }
    return total / (this->width * this->height);
}

double PixelSegment::perimeter() {
    double total = 0;
    auto len = this->points.size();
    for(int i = 0; i < len; ++i) {
        // is it a boundary point?
        cv::Point point = this->points[i];
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

double PixelSegment::area() { return this->points.size(); }

double PixelSegmentation::achievable_segmentation_accuracy(PixelSegmentation& ground_truth) {
    int total = 0;
    for(auto i = this->segments.begin(); i != this->segments.end(); i++) {
        // find intersecting gts and add the largest interesection
        PixelSegment this_seg = *i;

        int max = 0;
        for(auto j = ground_truth.segments.begin(); j != ground_truth.segments.end(); j++) {
            PixelSegment gt_seg = *j;
            if (!this_seg.bbox_intersect(gt_seg)) { continue; }
            intersection_result result = this_seg.intersection_and_diff(gt_seg);
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
    auto lab = convert_rgb_to_lab(image);
    
    vector<Vec3d> means = {};

    // compute the average colour for each face.
    for(auto &segment: segments) {
        Vec3d mean = {0,0,0};
        for(auto &p: segment.points) {
            mean += lab(p);
        }
        mean /= segment.area();
        means.push_back(mean);
    }
    
    double total_error = 0;


    for(int p_y = 0; p_y < lab.rows; ++p_y) {
        auto Mi = lab[p_y];
        for(int p_x = 0; p_x < lab.cols; ++p_x) {
            int label = this->segmentation_data.at<int>(p_y, p_x);
            Vec3d mean = means[label-1];
            Vec3d err = Mi[p_x] - mean;
            err = err.mul(err);

            //std::cout << pixel_error << std::endl;
            total_error += err[0] + err[1] + err[2];
        }
    }

    return total_error;
}

unsigned long PixelSegmentation::number_segments() {
    return this->segments.size();
}



