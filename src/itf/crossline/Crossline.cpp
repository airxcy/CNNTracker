//
//  Crossline.cpp
//  ITF_Inegrated
//
//  Created by Kun Wang on 9/22/2015.
//  Copyright (c) 2015 CUHK. All rights reserved.
//

#include "itf/crossline/Crossline.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/contrib/contrib.hpp>

#include "itf/crossline/flowIOOpenCVWrapper.h"
#include "itf/util/Util.hpp"

namespace itf {

void Crossline::Init(const std::string& pretrained_net_param, const std::string& feature_extraction_proto_file, const cv::Mat& pers_map, int rows, int cols) {
  itf::Engine::SetDevice(0);
  itf::Engine::set_mode(itf::Engine::GPU);

  feature_extraction_net_.reset(new itf::Net<float>(feature_extraction_proto_file, itf::TEST));
  feature_extraction_net_->CopyTrainedLayersFrom(pretrained_net_param);

  md_layer_ = boost::dynamic_pointer_cast <itf::MemoryDataLayer<float> >(feature_extraction_net_->layers()[0]);
  CHECK (md_layer_) << "The first layer is not a MemoryDataLayer!";

  rows_ = rows;
  cols_ = cols;

  pers_map.copyTo(pers_map_);
  CHECK_EQ(cv::Size(cols_, rows_), pers_map_.size()) << "Sizes of input arguments do not match";

  rotate_ = (cv::Mat_<float>(2, 2) << 0, -1, 1, 0);
}

void Crossline::Process(const cv::Mat& prevImg, const cv::Mat& nextImg, float** density_feature, float** x_feature, float** y_feature) {
  std::vector<cv::Mat> src1_spl;
  std::vector<cv::Mat> src2_spl;

  split(prevImg, src1_spl);
  split(nextImg, src2_spl);

  std::vector<cv::Mat> channels;
  channels.push_back(src1_spl[0]);
  channels.push_back(src1_spl[1]);
  channels.push_back(src1_spl[2]);
  channels.push_back(src2_spl[0]);
  channels.push_back(src2_spl[1]);
  channels.push_back(src2_spl[2]);
  cv::Mat input_data;
  merge(channels, input_data);

  std::vector<cv::Mat> dv (1, input_data);
  std::vector<int> dv1 (1, 0);

  md_layer_->AddMatVector(dv, dv1);

  feature_extraction_net_->ForwardFromTo(0, feature_extraction_net_->layers().size() - 1);
  const shared_ptr<Blob<float> > feature_blob_density = feature_extraction_net_->blob_by_name("density_out");
  const shared_ptr<Blob<float> > feature_blob_X = feature_extraction_net_->blob_by_name("vx_out");
  const shared_ptr<Blob<float> > feature_blob_Y = feature_extraction_net_->blob_by_name("vy_out");

  *density_feature = feature_blob_density->mutable_cpu_data();
  *x_feature = feature_blob_X->mutable_cpu_data();
  *y_feature = feature_blob_Y->mutable_cpu_data();
}

cv::Mat Crossline::VisualizeDensity(float* data, double alpha, double beta) {
  cv::Mat density_out = cv::Mat(rows_, cols_, CV_32F, data);
  density_out.convertTo(density_out, CV_8UC1, alpha, beta);
  cv::Mat heat;
  cv::applyColorMap(density_out, heat, cv::COLORMAP_JET); 
  return heat;
}

cv::Mat Crossline::VisualizeFlow(float* dataX, float* dataY) {
  cv::Mat xy[2];
  xy[0] = cv::Mat(rows_, cols_, CV_32F, dataY);
  xy[1] = cv::Mat(rows_, cols_, CV_32F, dataX);
  std::vector<cv::Mat> channels;
  channels.push_back(xy[0]);
  channels.push_back(xy[1]);
  cv::Mat flow;
  merge(channels, flow);

  cv::Mat bgr = flowToColor(flow);
  return bgr;
}

cv::Mat Crossline::Slice(const cv::Mat& img, cv::Point2i p1, cv::Point2i p2) {
  const int connectivity = 8;
  cv::LineIterator it(img, p1, p2, connectivity);
  std::vector<float> buf (it.count);

  for(int i=0; i<it.count; i++, it++)
    buf[i] = *(float *)*it;

  return cv::Mat(buf).clone();
}

std::vector<float> Crossline::CalcPredict(cv::Point2i p1, cv::Point2i p2, float* density, float* x, float* y) {
  // density_map
  cv::Mat density_map = cv::Mat(rows_, cols_, CV_32F, density).clone() * 0.001;
  cv::Mat cDensityPrd = Slice(density_map, p1, p2);
  cDensityPrd = cv::max(cDensityPrd, 0.0f);
  cDensityPrd = cv::min(cDensityPrd, 1.0f);

  // x
  cv::Mat vx_map = cv::Mat(rows_, cols_, CV_32F, x).clone() * 0.01;
  vx_map = vx_map.mul(pers_map_);
  cv::Mat cVelocityXPrd = Slice(vx_map, p1, p2);
  cVelocityXPrd = cv::max(cVelocityXPrd, -10);
  cVelocityXPrd = cv::min(cVelocityXPrd, 10);

  // y
  cv::Mat vy_map = cv::Mat(rows_, cols_, CV_32F, y).clone() * 0.01;
  vy_map = vy_map.mul(pers_map_);
  cv::Mat cVelocityYPrd = Slice(vy_map, p1, p2);
  cVelocityYPrd = cv::max(cVelocityYPrd, -10);
  cVelocityYPrd = cv::min(cVelocityYPrd, 10);

  cv::Mat vectorLine = (cv::Mat_<float>(1, 2) << (p2-p1).x, (p2-p1).y) / sqrt((p2-p1).x * (p2-p1).x + (p2-p1).y * (p2-p1).y);
  cv::Mat vnorm = rotate_ * vectorLine.t();
  cv::Mat velocityProj = (cVelocityXPrd * vnorm.at<float>(0) + cVelocityYPrd * vnorm.at<float>(1));

  cv::Mat left = (velocityProj > 0) / 255;
  left.convertTo(left, CV_32F);
  cv::Mat right = (velocityProj < 0) / 255;
  right.convertTo(right, CV_32F);

  cv::Mat leftPrd = cDensityPrd.mul(velocityProj.mul(left));
  cv::Mat rightPrd = cDensityPrd.mul(velocityProj.mul(right));
  std::vector<float> predicted;
  predicted.push_back(cv::sum(leftPrd)[0]);
  predicted.push_back(cv::sum(rightPrd)[0]);
  return predicted;
}

}  // namespace itf
