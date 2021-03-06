#include <opencv2/core/core.hpp>
#include <cstdint>

using namespace std;

#ifndef SEGMENTATION_H 
#define SEGMENTATION_H

class Segmentation {
    public:
        double boundary_recall(Segmentation& ground_truth);
        double boundary_recall(Segmentation& ground_truth, int epsilon);
        double undersegmentation_error(Segmentation& ground_truth);
        double achievable_segmentation_accuracy(Segmentation& ground_truth);
        double reconstruction_error(const cv::Mat& image);
        /// Determine the area of the intersection of the face with <label> with the pixel at (x,y).
        double intersection_area(int label, int x, int y);
        // /// Get the number of regions in this segmentation.
        unsigned long number_segments();
        // /// The pixel width of the represented image.
        // int size();

        int width, height;
};

class PixelSegmentation;

class PixelSegment {
    public:
        PixelSegmentation& segmentation;
        int32_t label;
        int x1;
        int x2;
        int y1;
        int y2;
        vector<cv::Point> points;
        int perimeter();
        int area();
        bool bbox_intersect(PixelSegment& other);
        int intersection(PixelSegment& other);
        PixelSegment(PixelSegmentation& seg);
};

struct intersection_metrics_result {
    double asa;
    double cue;
    double sue;
};

class PixelSegmentation : public Segmentation {
    public:
        static PixelSegmentation load_from_png(string& filename);
        static PixelSegmentation load_from_file(ifstream& file);
        void output_to_file(ofstream& out_file);
        double reconstruction_error(const cv::Mat& image);
        double compactness();
        double boundary_recall(PixelSegmentation& ground_truth);
        double boundary_recall(PixelSegmentation& ground_truth, int epsilon);
        intersection_metrics_result intersection_based_metrics(PixelSegmentation& ground_truth);
        double normalised_reconstruction_error(double recon_error);
    
        const inline int label_at(int i, int j);
        cv::Mat_<uchar> get_boundary_pixels() const;
        void initialise_segments();
        void compute_mean(cv::Mat& image, cv::Mat& output);

        PixelSegmentation(int32_t* data, uchar* b_data, int width, int height);
        PixelSegmentation(cv::Mat_<int32_t>& data, cv::Mat_<uchar>& boundary_data);
        PixelSegmentation();
    void draw_boundaries(cv::Mat& image, cv::Mat& output, bool thick_line=false, cv::Vec3b colour={0,0,255});
    
        unsigned long number_segments();
    
        cv::Mat_<int32_t> segmentation_data;
        cv::Mat_<uchar> boundary_data;
        vector<PixelSegment> segments;

};

class MeshSegmentation : Segmentation {
    public:
        void load_from_file(ifstream& file);
        void output_to_file(ofstream& out_file);

        MeshSegmentation();

};
#endif
