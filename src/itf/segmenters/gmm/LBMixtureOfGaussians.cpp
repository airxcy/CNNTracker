//
//  Copyright 2015 CUHK
//

#include "itf/segmenters/gmm/LBMixtureOfGaussians.hpp"

namespace itf {

LBMixtureOfGaussians::~LBMixtureOfGaussians() {
    delete m_pBGModel_;
};

void LBMixtureOfGaussians::process(const cv::Mat &img_input, cv::Mat &img_foreground, cv::Mat &img_background) {
  if(img_input.empty())
    return;
  
  IplImage *frame = new IplImage(img_input);
 
  if(firstTime_) {
    int w = cvGetSize(frame).width;
    int h = cvGetSize(frame).height;

    m_pBGModel_ = new BGModelMog(w,h);
    m_pBGModel_->InitModel(frame);

    m_pBGModel_->setBGModelParameter(0,sensitivity_);
    m_pBGModel_->setBGModelParameter(1,bgThreshold_);
    m_pBGModel_->setBGModelParameter(2,learningRate_);
    m_pBGModel_->setBGModelParameter(3,noiseVariance_);

    firstTime_ = false;
  }
  
  m_pBGModel_->UpdateModel(frame);

  img_foreground = cv::Mat(m_pBGModel_->GetFG());
  img_background = cv::Mat(m_pBGModel_->GetBG());

  delete frame;
};

void LBMixtureOfGaussians::SetParameters(const std::string& configure_file) {
    itf::GmmParameter gp;
    CHECK(itf::Util::ReadProtoFromTextFile(configure_file.c_str(), &gp)) << "Cannot read .prototxt file!";
    
    sensitivity_ = gp.sensitivity();
    bgThreshold_ = gp.bgthreshold();
    learningRate_ = gp.learningrate();
    noiseVariance_ = gp.noisevariance();
}


} // namespace itf
