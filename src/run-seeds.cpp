//
//  run-seeds.cpp
//  superpixel-benchmarks
//
//  Created by Donald S. F. Harvey on 29/11/2015.
//
//

#include "seeds.h"

std::string get_base_filename(const char* filename)
{
    using namespace std;
    string fName(filename);
    size_t pos = fName.rfind(".");
    if(pos == string::npos)  // No extension.
        return fName;

    if(pos == 0)    //. is at the front. Not an extension.
        return fName;

    return fName.substr(0, pos);
}

int main(int argc, char *argv[]) {
    using namespace std;
    using namespace cv::ximgproc;
    /* Load the image and convert to Lab colour space. */
    string root_name = get_base_filename(argv[1]);

    Mat image;
    image = imread(argv[1], CV_LOAD_IMAGE_COLOR);

    auto res = run_seeds(image, atoi(argv[2]));

    cout << res.number_segments() << endl;

//    cout << seeds->getNumberOfSuperpixels() << endl;

    Mat boundary_image = image.clone();

    uchar* contoursOut = NULL;
    int32_t* regionsOut = NULL;

    contoursOut = new uchar[image.cols * image.rows];
    regionsOut = new int32_t[image.cols * image.rows];

    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            contoursOut[i*image.cols + j] = res.boundary_data.at<uchar>(i, j);
            regionsOut[i*image.cols + j] = int32_t(res.segmentation_data.at<int>(i, j));
            if (res.boundary_data.at<uchar>(i, j)) {
                boundary_image.at<Vec3b>(i, j) = Vec3b(0, 0, 255);
            }
        }
    }

    imwrite(root_name + "_boundaries_" + argv[2] + ".png", boundary_image);

    cv::Mat means_image(image.rows, image.cols, image.type());
    res.compute_mean(image, means_image);

    imwrite(root_name + "_means_" + argv[2] + ".png", means_image);

    ofstream outFile;
    outFile.open(root_name + "_seg_" + argv[2] + ".dat", ios::binary | ios::out);

    outFile.write((char*)&(image.cols), sizeof(int32_t));
    outFile.write((char*)&(image.rows), sizeof(int32_t));

    for (int i=0; i < image.cols*image.rows; i++) {
        outFile.write((char*)&(regionsOut[i]), sizeof(int32_t));
    }

    for (int i=0; i < image.cols*image.rows; ++i) {
        outFile.write((char*)&(contoursOut[i]), sizeof(uchar));
    }
    outFile.close();

}
