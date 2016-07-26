//
//  isegmenter.hpp
//  ITF_Inegrated
//
//  Created by Xin Zhu on 8/18/2015.
//  Copyright (c) 2015 CUHK. All rights reserved.
//

#ifndef ISEGMENTER_HPP
#define ISEGMENTER_HPP

#include "itf/common.hpp"
#include "itf/proto/itf.pb.h"

namespace itf {

/**
 * @brief The abstract interface class for defining common interfaces of segmenter
 *
 */

class ISegmenter {
 public:
    /**
      * @brief Perform a segmention on frame or image.
      *
      * @param img_input input frame or image.
      * @param img_foreground output frame or image has been segmented.
      * @param img_background NULL.
      *
      */
    virtual void process(const cv::Mat &img_input, cv::Mat &img_foreground, cv::Mat &img_background) = 0;

    /**
      * @brief Set parameter for FCNN segmenter.
      *
      * @param configure_file 
      *    file name specifying the location of parameter file.
      *
      */
    virtual void SetParameters(const std::string& configure_file) = 0;

    virtual ~ISegmenter() {}
};


}  // namespace itf

#endif
