//
//  Copyright 2015 CUHK
//

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>

#include "itf/extracters/extracter_factory.hpp"

#include "itf/util/Util.hpp"


int main(int argc, const char * argv[]) {
    itf::ExtracterParameter ep;
    // Read configuration file
    if (!itf::Util::ReadProtoFromTextFile("./config/density_extracter.prototxt", &ep)) {
        std::cerr << "Cannot read .prototxt file!" << std::endl;
        return -1;
    } else {
        std::cout << "parse successfully" << std::endl;
    }

    // Create extracter factory
    itf::CExtracterFactory ef;
    // Factory instantiates an object of the specific type of extracter
    itf::IExtracter *iextracter = ef.SpawnExtracter(itf::CExtracterFactory::Density);
    iextracter->SetExtracterParameters(ep);

    // Necessary parameters to initialize extracter.
    // You can read them from configure file or hard coded them.
    std::string home_path(std::getenv("HOME"));

    int rows = 576, cols = 720;
    std::string pmap_path = home_path + "/forCPP/104207/pMap_matlab.csv";
    std::string roi_path = home_path + "/forCPP/104207/104207_roi.csv";

    if (ep.has_rows() && ep.has_cols()) {
        rows = ep.rows();
        cols = ep.cols();
    }
    if (ep.has_perspective_map()) {
        pmap_path = home_path + ep.perspective_map();
    }
    if (ep.has_roi()) {
        roi_path = home_path + ep.roi();
    }

    iextracter->SetImagesDim(rows, cols);
    iextracter->LoadPerspectiveMap(pmap_path);
    iextracter->LoadROI(roi_path);

    itf::Util util;
    // Get the perspective map and square it to generate a better heat map
    cv::Mat pmap = util.ReadPMAPtoMAT(pmap_path);
    pmap = pmap.mul(pmap);
    // Uncomment the following to load trained linear model
    // util.LoadLinearModel("/your/path/to/lm.csv");

    // Setup timer
    struct timespec tstart = {0, 0},  tend = {0, 0};

    std::string path = home_path + "/forCPP/104207/test_ori/";
    int train_num = 80;
    for (int index = 60; index < train_num; index++) {
        // Read each frame
        cv::Mat src = cv::imread(path + std::to_string(index+1) + ".jpg");
        if (src.empty()) {
            std::cerr << "Invalid Image" << std::endl;
            continue;
        }

        clock_gettime(CLOCK_MONOTONIC, &tstart);
        // Extract density feature from a frame loaded above
        std::vector<float> feature = iextracter->ExtractFeatures(src);
        clock_gettime(CLOCK_MONOTONIC, &tend);

        cv::Mat density_map(rows, cols, CV_32F, feature.data());

        // Output density of crowds on per pixel, if you want to get the exact number of people on a
        // specific frame, you need to employ a linear regressioner to get the final value of number of people.
        // cv::sum(A) sums along ALL dimensions and returns a single number (scalar).
        std::cout << cv::sum(density_map)[0] << std::endl;

        // Uncomment the following to employ trained linear model to predict value.
        // std::cout<< "counting number = " << util.Predict(cv::sum(density_map)[0]) << std::endl;

        std::cout <<(((double)tend.tv_sec + 1.0e-9*tend.tv_nsec) - ((double)tstart.tv_sec + 1.0e-9*tstart.tv_nsec)) * 1000 <<"ms"<< std::endl;

        // Generate a heat map
        // GenerateHeatMap(const cv::Mat& input, const cv::Mat& perspective_map, double alpha = 85.0, double beta = -6.0)
        cv::Mat heat = util.GenerateHeatMap(density_map, pmap);
        cv::imshow("test", heat);
        cv::waitKey(50);
    }

    // Free memory
    delete(iextracter);

    return 0;
}
