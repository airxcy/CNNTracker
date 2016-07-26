//
//  Crossline.hpp
//  ITF_Inegrated
//
//  Created by Kun Wang on 9/22/2015.
//  Copyright (c) 2015 CUHK. All rights reserved.
//

#ifndef CROSSLINE_HPP_
#define CROSSLINE_HPP_

#include <opencv2/core/core.hpp>

#include "itf/engine/net.hpp"
#include "itf/engine/data_layers.hpp"

namespace itf {

class Crossline {
 public:
  void Init(const std::string& pretrained_net_param, const std::string& feature_extraction_proto_file, const cv::Mat& pers_map, int rows, int cols);
  void Process(const cv::Mat& prevImg, const cv::Mat& nextImg, float** density_feature, float** x_feature, float** y_feature);
  /**
   * @brief Returns a vector that contains predicted values in two directions
   *
   * @param p1 The start point of the line
   * @param p2 The end point of the line
   * @param density The pointer to density_feature from Crossline::Process();
   * @param x The pointer to x_feature from Crossline::Process();
   * @param y The pointer to y_feature from Crossline::Process();
   *
   */
  std::vector<float> CalcPredict(cv::Point2i p1, cv::Point2i p2, float* density, float* x, float* y);
  cv::Mat Slice(const cv::Mat& img, cv::Point2i p1, cv::Point2i p2);
  cv::Mat VisualizeDensity(float* data, double alpha = 200.0, double beta = 0.0);
  cv::Mat VisualizeFlow(float* dataX, float* dataY);

 private:
  shared_ptr<Net<float> > feature_extraction_net_;
  boost::shared_ptr<itf::MemoryDataLayer<float> > md_layer_;
  int rows_;
  int cols_;
  cv::Mat pers_map_;
  cv::Mat rotate_;
};

}  // namespace itf

#endif  // CROSSLINE_HPP_
