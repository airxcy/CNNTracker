//
// Copyright 2015 CUHK
//

#include "itf/util/Util.hpp"

namespace itf {

cv::Mat Util::GenerateHeatMap(const cv::Mat& input, const cv::Mat& perspective_map, double alpha, double beta) {
    cv::Mat tmp_input = input.mul(perspective_map);
    // tmp_input = tmp_input * alpha + beta
    tmp_input.convertTo(tmp_input, CV_8UC1, alpha, beta);
    cv::Mat heat;
    cv::applyColorMap(tmp_input, heat, cv::COLORMAP_JET);
    return heat;
}

std::vector<std::pair<float, float> > Util::ReadPairToVec(const std::string& filename) {
    CvMLData mlData;
    mlData.read_csv(filename.c_str());
    const CvMat* tmp = mlData.get_values();
    cv::Mat cvPair(tmp, true);

    std::vector<std::pair<float, float> > pairs_vec;
    for (int i = 0; i < cvPair.rows; i+=2)
        pairs_vec.push_back(std::make_pair(cvPair.at<float>(i, 1), cvPair.at<float>(i+1, 1)));

    return pairs_vec;
}

bool Util::GeneratePerspectiveMap(std::vector<std::pair<float, float> > lines, int rows, int cols, const string& save_path_name) {
    CHECK_GE(lines.size(), 2) << "There must be 2 lines at least!";

    float threshold = 2.9412;  // This value is from Matlab.
    float height = 1.7;  // The default people hgiht is 1.7m
    int p_index = 0;
    cv::Mat pMap = cv::Mat::zeros(rows, cols, CV_32F);

    for (size_t i = 0; i < lines.size() - 1; ++i) {
        for (size_t j = i + 1; j < lines.size(); ++j) {
            ++p_index;
            float center_y_1 = (lines[i].first + lines[i].second) / 2;
            float center_y_2 = (lines[j].first + lines[j].second) / 2;
            float p_1 = std::abs(lines[i].first - lines[i].second) / height;
            float p_2 = std::abs(lines[j].first - lines[j].second) / height;
            float iter_now = std::abs(p_1 - p_2) / std::abs(center_y_1 - center_y_2);
            float p_max = (rows - center_y_1) * iter_now + p_1;
            cv::Mat pMap_temp = cv::Mat::zeros(rows, cols, CV_32F);
            for (int ii = rows - 1; ii > -1; --ii) {
                float* curr_row = pMap_temp.ptr<float>(ii);
                for (int jj = 0; jj < cols; ++jj) {
                    curr_row[jj] = p_max - iter_now * (rows - ii + 1);
                }
            }
            pMap = pMap + pMap_temp;
        }
    }
    cv::Mat pMap_out = pMap / p_index;
    pMap_out = cv::max(pMap_out, threshold);
    std::ofstream myfile(save_path_name);
    if (myfile.is_open()) {
        myfile << cv::format(pMap_out, "csv");
        myfile.close();
        return true;
    } else {
        std::cout << "Unable to open perspective file" << std::endl;
        return false;
    }
}

bool Util::GenerateROI(std::vector<std::pair<float, float> > points, const string& save_path_name) {
    CHECK_GE(points.size(), 3) << "There must be 3 lines at least!";

    std::ofstream myfile(save_path_name);
    if (myfile.is_open()) {
        for (size_t i = 0; i < points.size(); ++i)
            myfile << points[i].first << "," << points[i].second << std::endl;
        myfile.close();
        return true;
    } else {
        std::cout << "Unable to open roi file" << std::endl;
        return false;
    }
}

std::vector<double> Util::TrainLinearModel(std::vector<double> gts, std::vector<double> features,
    const std::string& save_name, double lambda) {
    arma::mat predictors(features);
    arma::vec responses(gts);

    mlpack::regression::LinearRegression lr(predictors.t(), responses, lambda);

    // Get the parameters, or coefficients.
    arma::vec parameters = lr.Parameters();
    std::vector<double> model;
    for (int i = 0; i < parameters.size(); ++i) {
        model.push_back(parameters[i]);
    }
    mlpack::data::Save(save_name, parameters);
    return model;
}

bool Util::LoadLinearModel(const std::string& model_path) {
    arma::mat model;
    arma::mat model_vec;
    mlpack::data::Load(model_path, model);
    model_vec = model.col(0);
    lr_.Parameters() = model_vec;
    return true;
}

double& Util::Predict(const double &input) {
    // Employ trained linear model to predict value.
    std::vector<double> d_feature;
    d_feature.push_back(input);
    arma::mat featureArma(d_feature);
    arma::vec predictArma;
    lr_.Predict(featureArma, predictArma);
    return predictArma[0];
}

cv::Mat Util::ReadROItoMAT(const std::string& filename, int rows, int cols) {
    CvMLData mlData;
    mlData.read_csv(filename.c_str());
    const CvMat* tmp = mlData.get_values();
    cv::Mat cvroi(tmp, true);

    std::vector< std::vector<cv::Point> > contours;
    std::vector<cv::Point> contour;
    for (int i = 0; i < cvroi.rows; ++i)
        contour.push_back(cv::Point(cvroi.at<float>(i, 0), cvroi.at<float>(i, 1)));
    contours.push_back(contour);
    // CV_32F is set due to elements-wise multification between perspective map and ROI.
    cv::Mat roi_mask = cv::Mat::zeros(cv::Size(cols, rows), CV_32F);
    // If it is negative, all the contours are drawn.
    int contourIdx = -1;
    // after invoking drawContours, roi_mask_ only contains two values of zero and one.
    cv::drawContours(roi_mask, contours, contourIdx, cv::Scalar(1), CV_FILLED);
    return roi_mask;
}

cv::Mat Util::ReadPMAPtoMAT(const std::string& filename) {
    CvMLData mlData;
    mlData.read_csv(filename.c_str());
    const CvMat* tmp = mlData.get_values();
    return cv::Mat(tmp, true);
}

}  // namespace itf
