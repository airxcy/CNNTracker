//
//  FCNN_Segmenter.hpp
//  ITF_Inegrated
//
//  Created by Xin Zhu on 8/18/2015.
//  Copyright (c) 2015 CUHK. All rights reserved.
//

#ifndef FCNN_SEGMENTER_HPP_
#define FCNN_SEGMENTER_HPP_

#include "itf/common.hpp"

#include "itf/segmenters/isegmenter.hpp"

#include "itf/engine/common.hpp"
#include "itf/engine/net.hpp"
#include "itf/engine/data_layers.hpp"
#include "itf/util/Util.hpp"

namespace itf {
// Concrete fcnn segmenter class

/**
 * @brief A type of segmenter serving as segment crowds on frame or image.
 *
 */
class FCNN_Segmenter : public ISegmenter {
 public:
     /**
      * @brief Perform a segmention on frame or image.
      *
      * @param img_input input frame or image.
      * @param img_output output frame or image has been segmented.
      * @param img_bgmodel NULL.
      *
      */
     void process(const cv::Mat &img_input, cv::Mat &img_output, cv::Mat &img_bgmodel);

     /**
      * @brief Set parameter for FCNN segmenter.
      *
      * @param configure_file
      *    file name specifying the location of parameter file.
      *
      */
     void SetParameters(const std::string& configure_file);

 private:
    shared_ptr<Net<float> > feature_extraction_net_;
    boost::shared_ptr<itf::MemoryDataLayer<float> > md_layer_;
    float scale_;
};


}  // namespace itf

#endif  // FCNN_SEGMENTER_HPP_
