//
//  Util.hpp
//  ITF_Inegrated
//
//  Created by Xin Zhu on 8/18/2015.
//  Copyright (c) 2015 CUHK. All rights reserved.
//

#ifndef UTIL_HPP_
#define UTIL_HPP_

#include <fcntl.h>
#include <fstream>

#include <google/protobuf/text_format.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

#include <mlpack/core.hpp>
#include <mlpack/methods/linear_regression/linear_regression.hpp>
#include <armadillo>

#include "itf/common.hpp"

namespace itf {

/**
 * @brief A  userful tool package that supplies many pre&post processing tools.
 *
 */
class Util {
 public:
    /**
    * @brief Returns heat map(score map)
    *
    * @param input The density map generated by CDensityExtracter::Extracter()
    * @param perspective_map The size of perspective map should match the size of the input, and to get a better result, please consider to square each element of it. (hint: perspective_map.mul(perspective_map))
    * @param alpha Optional scale factor
    * @param beta Optional delta added to the scaled values
    * 
    */
    cv::Mat GenerateHeatMap(const cv::Mat& input, const cv::Mat& perspective_map, double alpha = 85.0, double beta = -6.0);

    /**
    * @brief Returns the vector of pointer-pair, which is only used for Util::GeneratePerspectiveMap()
    *
    * @param filename The file name of the input csv file
    */
    std::vector<std::pair<float, float> > ReadPairToVec(const std::string& filename);

    /**
    * @brief Generates the perspective map and saves it as a csv file
    *
    * @param lines Each std::pair<head.y, foot.y> consists of the head's and the foot's y position of a person. Two persons at least. 
    * @param rows The number of rows of the generated perspective map
    * @param cols The number of columns of the generated perspective map
    * @param save_path_name The file name of the generated perspective map
    */
    bool GeneratePerspectiveMap(std::vector<std::pair<float, float> > lines, int rows, int cols, const string& save_path_name);
    
    /**
    * @brief Generates the ROI(Region of Interest) and saves it as a csv file
    *
    * @param points A vector of point<x, y> location used to draw the ROI
    * @param save_path_name The file name of the generated ROI
    */
    bool GenerateROI(std::vector<std::pair<float, float> > points, const string& save_path_name);

    /**
    * @brief  Returns a vector of coefficient estimates for a multilinear ridge regression of the responses in gts on the predictors in features
    *
    * @param gts A vector of ground truth
    * @param features A vector of observed responses
    * @param save_name The file name of the generated linear model
    * @param lambda Optional ridge parameter
    */
    std::vector<double> TrainLinearModel(std::vector<double> gts, std::vector<double> features, const std::string& save_name, double lambda = 0.1);

    /**
    * @brief Returns the ROI(cv::Mat) converted from a csv file
    *
    * @param filename The file name of the input csv file
    * @param rows The number of rows of ROI
    * @param cols The number of columns of ROI
    */
    cv::Mat ReadROItoMAT(const std::string& filename, int rows, int cols);

    /**
    * @brief Returns the perspective map(cv::Mat) converted from a csv file
    *
    * @param filename The file name of the input csv file
    */
    cv::Mat ReadPMAPtoMAT(const std::string& filename);

    /**
    * @brief Load an existing linear model
    *
    * @param model_path The file name of the input linear model(csv file)
    */
    bool LoadLinearModel(const std::string& model_path);
    double& Predict(const double &input);

    static bool ReadProtoFromTextFile(const char* filename, google::protobuf::Message* proto) {
        using google::protobuf::io::FileInputStream;
        using google::protobuf::TextFormat;
        int fd = open(filename, O_RDONLY);
        FileInputStream input(fd);
        return TextFormat::Parse(&input, proto);
    }

 private:
    mlpack::regression::LinearRegression lr_;
};

} // namespace itf

#endif  // UTIL_HPP_
