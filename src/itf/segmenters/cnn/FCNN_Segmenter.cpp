//
// Copyright 2015 CUHK
//

#include "itf/segmenters/cnn/FCNN_Segmenter.hpp"

#include <iostream>
#include <vector>

namespace itf {

void FCNN_Segmenter::process(const cv::Mat &img_input, cv::Mat &img_output, cv::Mat &img_bgmodel) {
    std::vector<cv::Mat> dv (1, img_input);
    std::vector<int> dv1 (1, 0);

    md_layer_->AddMatVector(dv, dv1);

    // Go through Caffe
    feature_extraction_net_->ForwardFromTo(0, feature_extraction_net_->layers().size() - 1);
    const shared_ptr<Blob<float> > feature_blob = feature_extraction_net_->blob_by_name("probs");

    float* feature_blob_data = feature_blob->mutable_cpu_data();
    img_output = cv::Mat(ceil(img_input.rows/scale_), ceil(img_input.cols/scale_), CV_32F, feature_blob_data);

    // Resize result
    cv::resize(img_output, img_output, cv::Size(img_input.cols, img_input.rows));
}

void FCNN_Segmenter::SetParameters(const std::string& configure_file) {
    itf::SegmenterParameter sp;
    CHECK(itf::Util::ReadProtoFromTextFile(configure_file.c_str(), &sp))<< "Cannot read .prototxt file!";

    itf::Engine::SetDevice(sp.device_id());
    itf::Engine::set_mode(itf::Engine::GPU);

    feature_extraction_net_.reset(new itf::Net<float>(sp.prototxt(), itf::TEST));
    feature_extraction_net_->CopyTrainedLayersFrom(sp.model());

    scale_ = sp.scale();

    // Check data layer
    md_layer_ = boost::dynamic_pointer_cast <itf::MemoryDataLayer<float> >(feature_extraction_net_->layers()[0]);
    CHECK (md_layer_) << "The first layer is not a MemoryDataLayer!";
}

}  // namespace itf
