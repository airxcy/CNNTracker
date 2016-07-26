//
//  segmenter_factory.hpp
//  ITF_Inegrated
//
//  Created by Xin Zhu on 8/18/2015.
//  Copyright (c) 2015 CUHK. All rights reserved.
//

#ifndef SEGMENTER_FACTORY_HPP_
#define SEGMENTER_FACTORY_HPP_

#include "itf/segmenters/isegmenter.hpp"

namespace itf {

/**
 * @brief Segmenter factory used to instantiate one of concrete segmenters
 *        such as FCNN Segmenter.
 *
 */
class CSegmenterFactory {
 public:
    /// The enum SegmenterType defines the available segmenter types
    enum SegmenterType {
        GMM,   //!< based on Gaussian Mixture model
        FCNN  //!< based on deep learning
    };
   /**
    *
    * @breif Instantiate one of concrete segmenters
    *
    * @param type 
    *    GMM or FCNN
    *
    */
    ISegmenter* SpawnSegmenter(const SegmenterType& type);
};


} // namespace itf


#endif  // SEGMENTER_FACTORY_HPP_
