//
//  Copyright 2015 CUHK
//

#include "itf/segmenters/segmenter_factory.hpp"
#include "itf/segmenters/gmm/LBMixtureOfGaussians.hpp"
#include "itf/segmenters/cnn/FCNN_Segmenter.hpp"


namespace itf {

// Segmenter factory class for generating the specific concrete segmenter class 
ISegmenter* CSegmenterFactory::SpawnSegmenter(const SegmenterType& type) {    
    ISegmenter *isegmenter = NULL;

    // Prawn a specific segmenter   
    switch(type) {
        case GMM:
            isegmenter = new LBMixtureOfGaussians();
            break;
        case FCNN:
            isegmenter = new FCNN_Segmenter();
            break;
        default:
            LOG(ERROR) << "error segmenter type";
    }

    return isegmenter;  
};


} // namespace itf
