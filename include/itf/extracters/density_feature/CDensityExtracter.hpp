//
//  CDensityExtracter.hpp
//  ITF_Inegrated
//
//  Created by Xin Zhu on 8/18/2015.
//  Copyright (c) 2015 CUHK. All rights reserved.
//

#ifndef CDENSITYEXTRACTER_HPP_
#define CDENSITYEXTRACTER_HPP_


#include <leveldb/db.h>

#include "itf/extracters/iextracter.hpp"

#include "itf/proto/itf.pb.h"
#include "itf/util/Util.hpp"

#include "itf/common.hpp"
#include "itf/engine/net.hpp"
#include "itf/engine/data_layers.hpp"

namespace itf {

/**
 * @brief A type of feature extracter serving as extracting
 *        crowds density on a frame or image.
 *
 * */
class CDensityExtracter : public IExtracter{
 public:
    // The cv::vector has all the available features by default.
    /**
     * @brief Returns a vector that contains a list of features.
     *
     * @param src The frame to be extract features
     *
     * This method is a concrete implement of interface of IExtracter.
     *
     */
    std::vector<float> ExtractFeatures(const cv::Mat &src);

    /**
     *
     * @brief Load Region of Interest.
     *
     * @param file file name specifying the location of ROI file.
     *
     */
    void LoadROI(const std::string &file);
    /**
     *
     * @brief Load perspective map.
     *
     * @param file file name specifying the location of perspective file.
     *
     */
    void LoadPerspectiveMap(const std::string &file);

    /**
     *
     * @brief Load parameters related to itf model.
     *
     * @param ep file name specifying the location of parameter file.
     *
     */
    void SetExtracterParameters(const itf::ExtracterParameter &ep);

    /**
     *
     * @brief Set the dimension of frame/image.
     *
     * @param rows Number of rows in a 2D array
     * @param cols Number of columns in a 2D array
     */
    inline void SetImagesDim(int rows, int cols) {
        CHECK_GT(rows, 0) << "rows must be greater than zero!";
        CHECK_GT(cols, 0) << "cols must be greater than zero!";
        rows_ = rows; cols_ = cols;
    }

 private:
    void SetPatchRange();
    void SetDensityIndex();

    // input
    int rows_ = 0;
    int cols_ = 0;

    int patch_width_ = 0;         // the size of patch might be returned from ExtractPatches()
    int patch_height_ = 0;
    int label_width_ = 0;         // the size of label might be 18 x 18
    int label_height_ = 0;
    std::string model_;
    std::string prototxt_;
    std::string extract_feature_blob_name_;
    float patch_para_ = 0;
    float overlap_ratio_;

    cv::Size patch_size_;
    std::vector<std::pair<cv::Point2f, double> > patches_;
    // patch_map_ is employed to store starting row index and starting and ending column index
    // thereon.
    std::map<int, std::vector<int> > patch_map_;
    double pad_width_ = 0;

    // warning: the template type should not be fixed-float. It is just for testing temporarily
    boost::shared_ptr<itf::Net<float> > feature_extraction_net_;
    boost::shared_ptr<itf::MemoryDataLayer<float> > md_layer_;

    cv::Mat perspective_map_;
    cv::Mat image_with_padding_;

    // intermediate to be used under process
    cv::Mat roi_mask_;
    cv::Mat roi_mask_8U_;

    cv::Mat density_image_all_index_;
};

}  // namespace itf


#endif  // CDENSITYEXTRACTER_HPP_
