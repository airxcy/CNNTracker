//
//  LBMixtureOfGaussians.hpp
//  ITF_Inegrated
//
//  Created by Xin Zhu on 8/18/2015.
//  Copyright (c) 2015 CUHK. All rights reserved.
//

#ifndef LBMIXTUREOFGAUSSIANS_HPP_
#define LBMIXTUREOFGAUSSIANS_HPP_

#include "itf/common.hpp"

#include "itf/segmenters/gmm/BGModelMog.hpp"
#include "itf/segmenters/isegmenter.hpp"
#include "itf/util/Util.hpp"

namespace itf {

class LBMixtureOfGaussians : public ISegmenter {
 public:
    ~LBMixtureOfGaussians();

    void process(const cv::Mat &img_input, cv::Mat &img_foreground, cv::Mat &img_background);
    void SetParameters(const std::string& configure_file);
 
 private:
    bool firstTime_ = true;
  
    BGModel* m_pBGModel_;
    int sensitivity_;
    int bgThreshold_;
    int learningRate_;
    int noiseVariance_;
};

} // namespace itf


#endif  // LBMIXTUREOFGAUSSIANS_HPP_
