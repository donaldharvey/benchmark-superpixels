#include "donald-slic.h"
#include <iostream>

using namespace cv;

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
    // Load the image and convert to Lab colour space.
    string root_name = get_base_filename(argv[1]);
    
    Mat image;
    image = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    
    Mat lab_image;
    cvtColor(image, lab_image, CV_BGR2Lab);
    
    auto res = run_slic(image, atoi(argv[2]), atoi(argv[3]));
    imwrite(root_name + "_means_" + argv[2] + "_" + argv[3] + ".png", res);
}
