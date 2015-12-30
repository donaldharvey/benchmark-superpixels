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

PixelSegmentation run_segmentation(string& algorithm_name, cv::Mat& image, int number_superpixels, map<string, string>& algo_opts) {
    if (algorithm_name == "slic") {
        if (algo_opts.count("compactness_weight")) {
            return run_slic(image, number_superpixels, stoi(algo_opts["compactness_weight"]));
        }
        return run_slic(image, number_superpixels, 10);
    }
    else if (algorithm_name == "seeds") {
        return run_seeds(image, number_superpixels);
    }
    else if (algorithm_name == "coarsetofine") {
        return run_coarsetofine(image, number_superpixels);
    }
    else {
        throw invalid_argument(algorithm_name + " is not implemented. Possible values are slic, seeds, coarsetofine.");
    }
}

PixelSegmentation run_segmentation(string& algorithm_name, cv::Mat& image, int number_superpixels) {
    map<string, string> algo_opts = map<string, string>();
    return run_segmentation(algorithm_name, image, number_superpixels, algo_opts);
}

struct bench_res {
    unsigned long number_segments;
    double br;
    double asa;
    double compactness;
    double reconstruction_error;
    double ue;
};

bench_res run_bench(PixelSegmentation& seg, cv::Mat& image, vector<string>& gts) {
    vector<double> brs;
    vector<double> asas;
    vector<double> ues;
    for(auto gt_p = gts.begin(); gt_p != gts.end(); ++gt_p) {
        ifstream gtfile;
        gtfile.open(*gt_p);
        PixelSegmentation gt = PixelSegmentation::load_from_file(gtfile);
        //gtfile.close();
        brs.push_back(seg.boundary_recall(gt, 2));
        asas.push_back(seg.achievable_segmentation_accuracy(gt));
        ues.push_back(seg.undersegmentation_error(gt));
    }
                              
    double best_br = *std::max_element(brs.begin(), brs.end());
    double best_asa = *std::max_element(asas.begin(), asas.end());
    double best_ue = *std::min_element(ues.begin(), ues.end());

    bench_res res = {
        .number_segments = seg.number_segments(), 
        .br = best_br, 
        .asa = best_asa, 
        .compactness = seg.compactness(), 
        .reconstruction_error = seg.reconstruction_error(image),
        .ue = best_ue
    };

    return res;
}

int main(int ac, char* av[]) {
    int number_superpixels;
    string algorithm_name;
    string output_filename;
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "produce help message")
        ("ground-truth,g", po::value< vector<string> >(), "ground truth dat paths")
        ("input-image", po::value<string>(), "input image path")
        ("number-superpixels,n", po::value<int>(&number_superpixels)->default_value(1000), "requested number of superpixels to input to the segmentation algorithm")
        ("algorithm", po::value<string>(&algorithm_name), "the name of the algorithm to use. possible values: slic, seeds")
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
       
    bool load_from_dat; 
    ifstream dat_file;
    if( (load_from_dat = vm.count("input") > 0) ) {
        dat_file.open(vm["input"].as<string>());
    }


    cv::Mat image = cv::imread(vm["input-image"].as<string>(), CV_LOAD_IMAGE_COLOR);
    PixelSegmentation seg = load_from_dat ? PixelSegmentation::load_from_file(dat_file) : run_segmentation(algorithm_name, image, number_superpixels);

    vector<string> gt_paths = vm["ground-truth"].as< vector<string> >();

    bench_res res = run_bench(seg, image, gt_paths);

    cout << "{\n"
    << "    \"number_segments\": " << res.number_segments << ",\n"
    << "    \"br\": " << res.br << ",\n"
    << "    \"asa\": " << res.asa << ",\n"
    << "    \"compactness\": " << res.compactness << ",\n"
    << "    \"reconstruction_error\": " << res.reconstruction_error << ",\n"
    << "    \"ue\": " << res.ue << "\n"
    << "}\n";
}
