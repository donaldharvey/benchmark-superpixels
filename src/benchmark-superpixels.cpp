#include "segmentation.h"
#include <iostream>
#include <fstream>
#include <opencv2/highgui/highgui.hpp>
#include <glob.h>
#include <vector>
#include <string>
#include <map>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <boost/progress.hpp>

#include "donald-slic.h"
#include "seeds.h"
#include "coursetofine.h"

namespace po = boost::program_options;
namespace fs = boost::filesystem;

using namespace std;

inline std::vector<std::string> glob(const std::string& pat){
    glob_t glob_result;
    glob(pat.c_str(),GLOB_TILDE,NULL,&glob_result);
    vector<string> ret;
    for(unsigned int i=0;i<glob_result.gl_pathc;++i){
        ret.push_back(string(glob_result.gl_pathv[i]));
    }
    globfree(&glob_result);
    return ret;
}

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
    else if (algorithm_name == "coursetofine") {
        return run_coursetofine(image, number_superpixels);
    }
    else {
        throw invalid_argument(algorithm_name + " is not implemented. Possible values are slic, seeds, coursetofine.");
    }
}

PixelSegmentation run_segmentation(string& algorithm_name, cv::Mat& image, int number_superpixels) {
    map<string, string> algo_opts = map<string, string>();
    return run_segmentation(algorithm_name, image, number_superpixels, algo_opts);
}

int main(int ac, char* av[]) {
//    try {
        int number_superpixels;
        string algorithm_name;
        string output_filename;
        po::options_description desc("Allowed options");
        desc.add_options()
            ("help", "produce help message")
            ("bsd-path,B", po::value<string>()->default_value("./BSDS500"), "location of the BSD images and datafiles")
            ("output-file,o", po::value<string>(), "where to put the output file")
            ("number-superpixels,n", po::value<int>(&number_superpixels)->default_value(1000), "requested number of superpixels to input to the segmentation algorithm")
            ("algorithm", po::value<string>(&algorithm_name), "the name of the algorithm to use. possible values: slic, seeds")
            ("input-dat-directory", po::value<string>(), "look for input .dats in this directory.")
        ;
        
        po::positional_options_description p;
        p.add("algorithm", 1);
        p.add("output-file", 1);
        
        po::variables_map vm;
        po::store(po::command_line_parser(ac, av).
                  options(desc).positional(p).run(), vm);
        po::notify(vm);
        
        if (vm.count("help")) {
            cout << "Usage: benchmark-superpixels [options] [algorithm-or-input-dat-directory] [output-file]" << endl;
            cout << desc << "\n";
            return 1;
        }
        
        fs::path bsd_path = fs::path(vm["bsd-path"].as<string>());
        assert(fs::is_directory(bsd_path));
        
        vector<string> possible_algorithms = {"slic", "seeds", "coursetofine"};
        bool load_from_files = vm.count("input-dat-directory") > 0;
        string input_dat_directory = "";
        if (load_from_files) {
            input_dat_directory = vm["input-dat-directory"].as<string>();
        }
        
        if(algorithm_name.length() && find(possible_algorithms.begin(), possible_algorithms.end(), algorithm_name) == possible_algorithms.end()) {
//            vm["load-from-files"] = algorithm_name;
            load_from_files = true;
            input_dat_directory = algorithm_name;
            cout << "Looking in " << input_dat_directory << " for .dat input files.\n";
            if (not fs::is_directory(fs::path(input_dat_directory))) {
                throw invalid_argument(input_dat_directory + " is not a directory");
            };
        }
        else {
            cout << "Using algorithm " << algorithm_name << ".\n";
        }
    
        
        vector<string> paths = glob((bsd_path / "*.jpg").string());
        ofstream csv;
        if (vm.count("output-file")) {
            output_filename = vm["output-file"].as<string>();
        }
        else {
            if (load_from_files) {
                output_filename = "benchmark_results.tsv";
            }
            else {
                output_filename = algorithm_name + "-" + to_string(number_superpixels) + ".tsv";
            }
        }
        csv.open(output_filename, ios::out);
        csv << "Name\tNumber superpixels\tBR\tASA\tCO\tRE\tUE\n";
    
        cout << "Running on " << paths.size() << " test images..." << endl;
        boost::progress_display show_progress( paths.size() );
        
        for(auto i = paths.begin(); i != paths.end(); ++i) {
            cv::Mat image = cv::imread(*i, CV_LOAD_IMAGE_COLOR);
            string basename = fs::basename(fs::path(*i));
            ifstream input_file;
            if (load_from_files) {
                string input_path = ((fs::path(input_dat_directory) / basename)).string() + ".dat";
                if(not fs::is_regular_file(input_path)) {
                    throw runtime_error("Missing .dat file, need " + basename + ".dat");
                }
                input_file.open(input_path, ios::binary);
            }
            PixelSegmentation seg = load_from_files ? PixelSegmentation::load_from_file(input_file) : run_segmentation(algorithm_name, image, number_superpixels);
            string gtglob = *i;
            auto f = i->find(".jpg");
            gtglob.replace(f, 4, "_[0-9].dat");
            vector<string> gts = glob(gtglob);
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
                
                cv::Mat means_image(image.rows, image.cols, image.type());
                gt.compute_mean(image, means_image);
                
                imwrite((bsd_path / basename).string() + "_" + to_string(gt_p - gts.begin()) + "_means.png", means_image);
            }
                              
            double best_br = *std::max_element(brs.begin(), brs.end());
            double best_asa = *std::max_element(asas.begin(), asas.end());
            double best_ue = *std::min_element(ues.begin(), ues.end());

            csv << basename << "\t" <<
            seg.number_segments() << "\t" <<
            best_br << "\t" <<
            best_asa << "\t" <<
            seg.compactness() << "\t" <<
            seg.reconstruction_error(image) << "\t" <<
            best_ue << endl;
            csv.flush();
            
            ++show_progress;
        }
        csv.close();
//    }
//    catch(exception& e) {
//        cerr << "error: " << e.what() << "\n";
//        return 1;
//    }
//    catch(...) {
//        cerr << "Exception of unknown type!\n";
//    }
    return 0;
}
