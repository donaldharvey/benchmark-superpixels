#include "segmentation.h"
#include <iostream>
#include <fstream>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include <string>
#include <map>

#include "donald-slic.h"
#include "seeds.h"
#include "coarsetofine.h"
#include <boost/program_options.hpp>

namespace po = boost::program_options;

using namespace std;

// PixelSegmentation run_segmentation(string& algorithm_name, cv::Mat& image, int number_superpixels, map<string, string>& algo_opts) {
//     if (algorithm_name == "slic") {
//         if (algo_opts.count("compactness_weight")) {
//             return run_slic(image, number_superpixels, stoi(algo_opts["compactness_weight"]));
//         }
//         return run_slic(image, number_superpixels, 10);
//     }
//     else if (algorithm_name == "seeds") {
//         return run_seeds(image, number_superpixels);
//     }
//     else if (algorithm_name == "coarsetofine") {
//         return run_coarsetofine(image, number_superpixels);
//     }
//     else {
//         throw invalid_argument(algorithm_name + " is not implemented. Possible values are slic, seeds, coarsetofine.");
//     }
// }

// PixelSegmentation run_segmentation(string& algorithm_name, cv::Mat& image, int number_superpixels) {
//     map<string, string> algo_opts = map<string, string>();
//     return run_segmentation(algorithm_name, image, number_superpixels, algo_opts);
// }

struct bench_res {
    unsigned long number_segments;
    double br2;
    double br1;
    double br0;
    double asa;
    double compactness;
    double reconstruction_error;
    double cue;
    double sue;
};

bench_res run_bench(PixelSegmentation& seg, cv::Mat& image, string& gt_path) {
    ifstream gtfile;
    gtfile.open(gt_path);
    PixelSegmentation gt = PixelSegmentation::load_from_file(gtfile);
    //gtfile.close();
    double br2 = seg.boundary_recall(gt, 2);
    double br1 = seg.boundary_recall(gt, 1);
    double br0 = seg.boundary_recall(gt, 0);
//    double asa = seg.achievable_segmentation_accuracy(gt);
//    double cue = seg.corrected_undersegmentation_error(gt);
//    double sue = seg.symmetric_undersegmentation_error(gt);
    double asa, cue, sue = 0;
    
    bench_res res = {
        .number_segments = seg.number_segments(), 
        .br2 = br2,
        .br1 = br1,
        .br0 = br0,
        .asa = asa,
        .compactness = seg.compactness(),
        .reconstruction_error = seg.reconstruction_error(image),
        .cue = cue,
        .sue = sue,
    };

    return res;
}

int main(int ac, char* av[]) {
    int number_superpixels;
    // string algorithm_name;
    string output_filename;
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "produce help message")
        ("ground-truth,g", po::value<string>(), "ground truth dat path")
        ("input-image", po::value<string>(), "input image path")
        ("number-superpixels,n", po::value<int>(&number_superpixels)->default_value(1000), "requested number of superpixels to input to the segmentation algorithm")
        // ("algorithm", po::value<string>(&algorithm_name), "the name of the algorithm to use. possible values: slic, seeds")
        ("input", po::value<string>(), "location of input dat")
    ;
        
    po::positional_options_description p;
    // p.add("algorithm", 1);
    // p.add("output-file", 1);
    
    po::variables_map vm;
    po::store(po::command_line_parser(ac, av).
              options(desc).positional(p).run(), vm);
    po::notify(vm);
    
    if (vm.count("help")) {
        cout << "Usage: benchmark-single [options]" << endl;
        cout << desc << "\n";
        return 1;
    }
       
    bool load_from_png;
    string png_file;
    if( (load_from_png = vm.count("input") > 0) ) {
        png_file = vm["input"].as<string>();
    }


    cv::Mat image = cv::imread(vm["input-image"].as<string>(), CV_LOAD_IMAGE_COLOR);
    PixelSegmentation seg = PixelSegmentation::load_from_png(png_file);

    string gt_path = vm["ground-truth"].as<string>();

    bench_res res = run_bench(seg, image, gt_path);
    
    cout.precision(10);

    cout << "{\n"
    << "    \"number_segments\": " << res.number_segments << ",\n"
    << "    \"br2\": " << res.br2 << ",\n"
    << "    \"br1\": " << res.br1 << ",\n"
    << "    \"br0\": " << res.br0 << ",\n"
    << "    \"asa\": " << res.asa << ",\n"
    << "    \"compactness\": " << res.compactness << ",\n"
    << "    \"reconstruction_error\": " << res.reconstruction_error << ",\n"
    << "    \"sue\": " << res.sue << "\n"
    << "    \"cue\": " << res.cue << "\n"
    << "}\n";
}
