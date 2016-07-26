//
//  Reid.hpp
//  ITF_Inegrated
//
//  Created by Kun Wang on 10/13/2015.
//  Copyright (c) 2015 CUHK. All rights reserved.
//

#ifndef ITF_INTEGRATED_REID_HPP_
#define ITF_INTEGRATED_REID_HPP_

#include <opencv2/core/core.hpp>

#include <string>
#include <vector>

#include "itf/engine/net.hpp"
#include "itf/engine/data_layers.hpp"

namespace itf {

class Reid {
 public:
  /**
   *
   * @brief Initialize the network
   *
   * @param pretrained_net_param file name of model
   * @param feature_extraction_proto_file file name of network
   *
   */
  void Init(const std::string& pretrained_net_param,
    const std::string& feature_extraction_proto_file);

  /**
   *
   * @brief Calculate and extract features, the returned pointer has a size of images.size() * 256
   *
   * @param images a vector of images to be processed
   * @param feature_blob_name specify the blob name(fc7_bn, by default) to be extracted
   *
   */
  float* CalcFeatures(const std::vector<cv::Mat>& images,
    const std::string& feature_blob_name = "fc7_bn") const;

 private:
  boost::shared_ptr<Net<float> > feature_extraction_net_;
  boost::shared_ptr<MemoryDataLayer<float> > md_layer_;
};

}  // namespace itf

#endif  // ITF_INTEGRATED_REID_HPP_