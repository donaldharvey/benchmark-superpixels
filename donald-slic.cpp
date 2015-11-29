#include "donald-slic.h"

using namespace cv;
#define NR_ITERATIONS 10

struct labxy {
    Vec3d color;
    Point2d point;
    friend std::ostream& operator<< (std::ostream& o, const labxy& l) {
        return o << "[" << l.color << "," << l.point << "]";
    }
};

Point find_local_minimum(Mat& image, Point center) {
    double min_grad = DBL_MAX;
    Point loc_min = Point(center.x, center.y);
    
    for (int i = center.x-1; i < center.x+2; i++) {
        for (int j = center.y-1; j < center.y+2; j++) {
            Vec3b c1 = image.at<Vec3b>(j+1, i);
            Vec3b c2 = image.at<Vec3b>(j, i+1);
            Vec3b c3 = image.at<Vec3b>(j, i);
            double i1 = c1.val[0];
            double i2 = c2.val[0];
            double i3 = c3.val[0];
            
            /* Compute horizontal and vertical gradients and keep track of the
             minimum. */
            if (sqrt(pow(i1 - i3, 2)) + sqrt(pow(i2 - i3,2)) < min_grad) {
                min_grad = fabs(i1 - i3) + fabs(i2 - i3);
                loc_min.x = i;
                loc_min.y = j;
            }
        }
    }
    
    return loc_min;
}

double compute_dist(labxy& px, labxy& center, double normalization) {
    double d_lab = norm(px.color - center.color);
    double d_xy = norm(px.point - center.point);
    double d = pow(d_lab, 2) + pow(normalization * d_xy, 2);
    return d;
}

unsigned long init_centers(std::vector<labxy>& centers, std::vector<int>& center_counts, Mat& image, int step, int w, int h) {
    int xstrips = (0.5+double(w)/double(step));
    int ystrips = (0.5+double(h)/double(step));
    int xerr = w - step*xstrips; if(xerr < 0){ xstrips--; xerr = w - step*xstrips;}
    int yerr = h - step*ystrips; if(yerr < 0){ ystrips--; yerr = h - step*ystrips;}
    double xerrperstrip = double(xerr)/double(xstrips);
    double yerrperstrip = double(yerr)/double(ystrips);
    int xoff = step/2;
    int yoff = step/2;
    
    for (int y = 0; y < ystrips; ++y) {
        int ye = y*yerrperstrip;
        for( int x = 0; x < xstrips; x++ ) {
            int xe = x*xerrperstrip;
            int seedx = x * step + xoff + xe;
            int seedy = y * step + yoff + ye;
            Point min = find_local_minimum(image, Point(seedx,seedy));
            Vec3d color = Vec3d(image.at<Vec3b>(min));
            labxy center = {.color = color, .point = Point2d(min)};
            centers.push_back(center);
            center_counts.push_back(0);
        }
    }
    
//    for (int i = step/2; i < h - step/2; i += step) {
//        for (int j = step/2; j < w - step/2; j += step) {
//            Point min = find_local_minimum(image, Point(j,i));
//            Vec3d color = Vec3d(image.at<Vec3b>(min));
//            labxy center = {.color = color, .point = Point2d(min)};
//            centers.push_back(center);
//            center_counts.push_back(0);
//        }
//    }
    return centers.size();
}

void run_iteration(std::vector<labxy>& centers, Mat& clusters, int number_centers, Mat& lab_image, int step, int w, int h, int m) {
    Rect bounds = Rect(0, 0, w, h);
    Mat distances(h,w,CV_64F, DBL_MAX);
    std::vector<int> center_counts(number_centers, 0);
    
    for(int j=0; j < number_centers; ++j) {
        labxy cen = centers[j];
        Rect roi = Rect(cen.point.x - step, cen.point.y - step, 2*step, 2*step) & bounds;
        for(int y = roi.y; y < roi.y + roi.height; y++) {
            for(int x = roi.x; x < roi.x + roi.width; x++) {
                Vec3d col = Vec3d(lab_image.at<Vec3b>(y,x));
                labxy px = {.color = col, .point = Point2d(x,y)};
                double d = compute_dist(px, cen, double(m)/double(step));
                if (d < distances.at<double>(y,x)) {
                    distances.at<double>(y,x) = d;
                    clusters.at<int>(y,x) = j;
                }
            }
        }
    }

    std::vector<labxy> means;

    for(int i=0; i < number_centers; ++i) {
        means.push_back(labxy({.point = Point2d(0,0), .color=Vec3d(0,0,0)}));
    }

    for(int y = 0; y < clusters.rows; y++)
    {
        const int* My = clusters.ptr<int>(y);
        const Vec3b* Iy = lab_image.ptr<Vec3b>(y);
        for(int x = 0; x < clusters.cols; x++) {
            int idx = My[x];
            means[idx].point += Point2d(x,y);
            means[idx].color += Iy[x];
            center_counts[idx] += 1;
        }
    }

    for(int i=0; i<number_centers; ++i) {
        if(center_counts[i] <= 0) { center_counts[i] = 1; }
        centers[i].point = means[i].point / center_counts[i];
        centers[i].color = means[i].color / center_counts[i];
    }
}

Mat enforce_connectivity(Mat& clusters, int number_centers) {
    int label = 0, adjlabel = 0;
    const int lims = (clusters.rows * clusters.cols) / number_centers;
    
    const int dx4[4] = {-1,  0,  1,  0};
    const int dy4[4] = { 0, -1,  0,  1};
    
    /* Initialize the new cluster matrix. */
    Mat new_clusters(clusters.rows,clusters.cols,CV_32S, -1);
    
    for (int i = 0; i < clusters.rows; i++) {
        for (int j = 0; j < clusters.cols; j++) {
            if (new_clusters.at<int>(i, j) == -1) {
                std::vector<Point> elements;
                elements.push_back(Point(j, i));
            
                /* Find an adjacent label, for possible use later. */
                for (int k = 0; k < 4; k++) {
                    int x = elements[0].x + dx4[k], y = elements[0].y + dy4[k];
                    
                    if (x >= 0 && x < clusters.cols && y >= 0 && y < clusters.rows) {
                        if (new_clusters.at<int>(y, x) >= 0) {
                            adjlabel = new_clusters.at<int>(y, x);
                        }
                    }
                }
                
                int count = 1;
                for (int c = 0; c < count; c++) {
                    for (int k = 0; k < 4; k++) {
                        int x = elements[c].x + dx4[k], y = elements[c].y + dy4[k];
                        
                        if (x >= 0 && x < clusters.cols && y >= 0 && y < clusters.rows) {
                            if (new_clusters.at<int>(y, x) == -1 && clusters.at<int>(i, j) == clusters.at<int>(y, x)) {
                                elements.push_back(Point(x, y));
                                new_clusters.at<int>(y, x) = label;
                                count += 1;
                            }
                        }
                    }
                }
                
                /* Use the earlier found adjacent label if a segment size is
                   smaller than a limit. */
                if (count <= lims >> 2) {
                    for (int c = 0; c < count; c++) {
                        new_clusters.at<int>(elements[c].y, elements[c].x) = adjlabel;
                    }
                    label -= 1;
                }
                label += 1;
            }
        }
    }

    return new_clusters;
}

Mat generate_boundaries(Mat& labels) {
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

PixelSegmentation run_slic(Mat& image, int target_superpixel_number, int m) {
    Mat lab_image;
    cvtColor(image, lab_image, CV_BGR2Lab);
    
    int w = lab_image.cols;
    int h = lab_image.rows;
    
    int step = int( sqrt(w*h / double(target_superpixel_number)) );
    
    std::vector<labxy> centers;
    std::vector<int> center_counts;
    Mat clusters(h,w,CV_32S, -1);
    
    int number_centers = int(init_centers(centers, center_counts, lab_image, step, w, h));
    
    for(int i=0; i < NR_ITERATIONS; ++i) {
        run_iteration(centers, clusters, number_centers, lab_image, step, w, h, m);
    }

//    std::cout << clusters;
    
    Mat labels = enforce_connectivity(clusters, number_centers) + 1;
//        std::cout << labels;
    Mat boundaries = generate_boundaries(labels);
//        std::cout << boundaries;
    return PixelSegmentation(labels, boundaries);

}

//std::string get_base_filename(const char* filename)
//{
//    using namespace std;
//    string fName(filename);
//    size_t pos = fName.rfind(".");
//    if(pos == string::npos)  // No extension.
//        return fName;
//    
//    if(pos == 0)    //. is at the front. Not an extension.
//        return fName;
//    
//    return fName.substr(0, pos);
//}

/* int main(int argc, char *argv[]) {
    using namespace std;
 Load the image and convert to Lab colour space.
    string root_name = get_base_filename(argv[1]);
    
    Mat image;
    image = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    
    Mat lab_image;
    cvtColor(image, lab_image, CV_BGR2Lab);
    
    auto res = run_slic(image, atoi(argv[2]), atoi(argv[3]));
    // // Display the contours and show the result.
    // slic.display_contours(image, CV_RGB(0,0,0));
    // cvShowImage("result", image);
    // cvWaitKey(0);
    // cvSaveImage(argv[4], image);
    
    Mat boundary_image = image.clone();
    
    uchar* contoursOut = NULL;
    int32_t* regionsOut = NULL;
    
    contoursOut = new uchar[image.cols * image.rows];
    regionsOut = new int32_t[image.cols * image.rows];
    
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            contoursOut[i*image.cols + j] = res.boundary_data.at<uchar>(i, j);
//            std::cout << "Wrote " << res.regions.at<int>(i, j) << std::endl;
            regionsOut[i*image.cols + j] = int32_t(res.segmentation_data.at<int>(i, j));
            if (res.boundary_data.at<uchar>(i, j)) {
                boundary_image.at<Vec3b>(i, j) = Vec3b(0, 0, 255);
            }
        }
    }
    
    imwrite(root_name + "_boundaries_" + argv[2] + "_" + argv[3] + ".png", boundary_image);
    
    cv::Mat means_image(image.rows, image.cols, image.type());
    res.compute_mean(image, means_image);
    
    imwrite(root_name + "_means_" + argv[2] + "_" + argv[3] + ".png", means_image);
    
    ofstream outFile;
    outFile.open(root_name + "_seg_" + argv[2] + "_" + argv[3] + ".dat", ios::binary | ios::out);
    
    outFile.write((char*)&(image.cols), sizeof(int32_t));
    outFile.write((char*)&(image.rows), sizeof(int32_t));
    
    for (int i=0; i < image.cols*image.rows; i++) {
        outFile.write((char*)&(regionsOut[i]), sizeof(int32_t));
    }
    
    for (int i=0; i < image.cols*image.rows; ++i) {
        outFile.write((char*)&(contoursOut[i]), sizeof(uchar));
    }
    outFile.close();
    
}*/
