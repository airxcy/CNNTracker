//
//  Reid.cpp
//  ITF_Inegrated
//
//  Created by Kun Wang on 10/13/2015.
//  Copyright (c) 2015 CUHK. All rights reserved.
//

#include "itf/reid/Reid.hpp"

#include <string>
#include <vector>

namespace itf {

void Reid::Init(const std::string& pretrained_net_param,
    const std::string& feature_extraction_proto_file) {
  // fix the GPU mode and device_id for now
  Engine::SetDevice(0);
  Engine::set_mode(Engine::GPU);
  // setup network
  feature_extraction_net_.reset(
    new Net<float>(feature_extraction_proto_file, TEST));
  feature_extraction_net_->CopyTrainedLayersFrom(pretrained_net_param);
  // check input layer, we prefer MemoryDataLayer for speed
  md_layer_ = boost::dynamic_pointer_cast <MemoryDataLayer<float> >(
    feature_extraction_net_->layers()[0]);
  CHECK(md_layer_) << "The first layer is not a MemoryDataLayer!";
}

float* Reid::CalcFeatures(const std::vector<cv::Mat>& images,
    const std::string& feature_blob_name) const {
  // initialize dummy labels, which is required by MemoryDataLayer
  std::vector<int> dv1(images.size(), 0);
  // Note: for sake of efficiency,
  // we don't check the input image size, which
  // should be cv::Size(56, 144) exactly, or the program will crash
  md_layer_->AddMatVector(images, dv1);
  // forward
  feature_extraction_net_->ForwardFromTo(
    0, feature_extraction_net_->layers().size() - 1);
  // extract feature blob
  return feature_extraction_net_->blob_by_name(
    feature_blob_name)->mutable_cpu_data();
}

}  // namespace itf