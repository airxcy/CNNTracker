//
// Copyright 2015 CUHK
//
// Author: Xin Zhu
// Date: 7 December 2014.

#include "itf/extracters/density_feature/CDensityExtracter.hpp"

// enum { CV_8U=0, CV_8S=1, CV_16U=2, CV_16S=3, CV_32S=4, CV_32F=5, CV_64F=6 };

namespace itf {

std::vector<float> CDensityExtracter::ExtractFeatures(const cv::Mat &src) {
    // Pre-porcess image
    src.copyTo(image_with_padding_, roi_mask_8U_);
    copyMakeBorder(image_with_padding_, image_with_padding_, pad_width_, pad_width_, 0, 0, cv::BORDER_REPLICATE);
    copyMakeBorder(image_with_padding_, image_with_padding_, 0, 0, pad_width_, pad_width_, cv::BORDER_REFLECT);

    // Extract patches
    std::vector<cv::Mat> patch_vec;
    for (size_t row = 0; row < patches_.size(); ++row) {
        double real_len = patches_[row].second*2+1;
        cv::Mat image_patch = image_with_padding_(cv::Rect(patches_[row].first.x - patches_[row].second, patches_[row].first.y - patches_[row].second, real_len, real_len)).clone();
        cv::resize(image_patch, image_patch, patch_size_);
        patch_vec.push_back(image_patch);
    }

    // Go through Engine
    md_layer_->AddMatVectorDensity(patch_vec);
    feature_extraction_net_->ForwardFromTo(0, feature_extraction_net_->layers().size() - 1);
    shared_ptr<itf::Blob<float> > feature_blob = feature_extraction_net_->blob_by_name(extract_feature_blob_name_);

    // Recover patches
    cv::Mat density_image_all_out = cv::Mat::zeros(rows_ + pad_width_ + pad_width_, cols_ + pad_width_ + pad_width_, CV_32F);
    for (size_t row = 0; row < patches_.size(); ++row) {
        double pers_value = patches_[row].second;
        float coff = 0.1 * pow(18 / (pers_value*2+1), 2);
        cv::Mat density_prediction = cv::Mat(label_height_, label_width_, CV_32F, feature_blob->mutable_cpu_data() + feature_blob->offset(row)) * coff;
        cv::resize(density_prediction, density_prediction, cv::Size(pers_value*2+1, pers_value*2+1));
        int current_col = patches_[row].first.x;
        int current_row = patches_[row].first.y;
        int row_index = 0;
        for (int row = current_row - pers_value; row < current_row + pers_value+1; ++row) {
            float* density_out = density_image_all_out.ptr<float>(row);
            float* density_index = density_image_all_index_.ptr<float>(row);
            float* density_pred = density_prediction.ptr<float>(row_index++);
            int col_index = 0;
            for (int col = current_col - pers_value; col < current_col + pers_value+1; ++col)
                density_out[col] += density_pred[col_index++] * density_index[col];
        }
    }
    cv::Mat vec = density_image_all_out(cv::Rect(pad_width_, pad_width_, cols_, rows_)).clone().reshape(0, 1);
    std::vector<float> vgradx;
    vgradx.reserve(vec.cols);
    memcpy(&(vgradx[0]), vec.data, vec.cols * sizeof(float));

    return vgradx;
}

void CDensityExtracter::LoadROI(const std::string &file) {
    Util util;
    roi_mask_ = util.ReadROItoMAT(file, rows_, cols_);

    roi_mask_.convertTo(roi_mask_8U_, CV_8U);

    cv::Mat perspective_map_temp;
    perspective_map_temp = perspective_map_.mul(roi_mask_);

    // max and min value of perspective weights in ROI.
    double min, max;
    cv::minMaxIdx(perspective_map_temp, &min, &max);

    pad_width_ = std::ceil(max*patch_para_)*2;

    SetPatchRange();
    SetDensityIndex();
}

void CDensityExtracter::LoadPerspectiveMap(const std::string &file) {
    Util util;
    perspective_map_ = util.ReadPMAPtoMAT(file);
    CHECK_EQ(perspective_map_.size(), cv::Size(cols_, rows_)) << "The size of perspectivea_map doesn't match predefined size!";
}

void CDensityExtracter::SetExtracterParameters(const itf::ExtracterParameter &ep) {
    // initialize various of parameters
    patch_width_ = ep.patch_width();
    patch_height_ = ep.patch_height();

    patch_size_ = cv::Size(patch_width_, patch_height_);

    label_width_ = ep.label_width();
    label_height_ = ep.label_height();

    model_ = ep.model();
    prototxt_ = ep.prototxt();
    extract_feature_blob_name_ = ep.extract_feature_blob_name();

    patch_para_ = ep.patch_para();
    overlap_ratio_ = 1.0f - ep.overlap_ratio();
    if (overlap_ratio_ > 1.0f || overlap_ratio_ < 0.0f)
        overlap_ratio_ = 0.5f;

    // Use GPU by default
    if (ep.solver_mode()) {
        itf::Engine::SetDevice(ep.device_id());
        itf::Engine::set_mode(itf::Engine::GPU);
    } else {
        itf::Engine::set_mode(itf::Engine::CPU);
    }
    // Initialize Caffe Network
    feature_extraction_net_.reset(new itf::Net<float>(ep.prototxt(), itf::TEST));
    feature_extraction_net_->CopyTrainedLayersFrom(ep.model());

    // Get pointer to MemoryDataLayer instance.
    md_layer_ = boost::dynamic_pointer_cast <itf::MemoryDataLayer<float>>(feature_extraction_net_->layers()[0]);
    CHECK(md_layer_) << "The first layer is not a MemoryDataLayer!";
}

void CDensityExtracter::SetPatchRange() {
    //
    // patch_map is a kind of std::map<key, value> whose key representing the row index on which patches will be
    // extract and whose value is type of integer vector representing the start column index and
    // end column index on the specific patching row.
    //
    // thus, patch_map actually contains the two-dimensional, which means rows and columns,
    // range of the activity of extracting patches.
    //
    // -------------------------------------------------------------
    // | Key(row index) | Value(start and end column index on row) |
    // -------------------------------------------------------------
    //
    std::vector<int> vec_row_index;
    // generate the start and end row for extracting patches.
    for (int row = 0; row < rows_; row++) {
        for (int col = 0; col < cols_; col++) {
            if (roi_mask_.at<float>(row, col) != 0) {
                vec_row_index.push_back(row);
                break;
            }
        }
    }

    // start row and end row for extracting patches
    int start_row_patch = vec_row_index.front();
    int end_row_patch = vec_row_index.back();

#if 1
    // new code for supporting non-convex roi region.
    int row = start_row_patch;
    while (row <= end_row_patch) {
        for (int col = 0; col < cols_; col++) {
            if (roi_mask_.at<float>(row, col) != 0 && roi_mask_.at<float>(row, (col + 1)) != 0) {
                if(roi_mask_.at<float>(row, (col - 1)) == 0) {
                    // start column index (011)
                    patch_map_[row].push_back(col);
                }
            } else if (roi_mask_.at<float>(row, col) != 0 && roi_mask_.at<float>(row, (col + 1)) == 0) {
                if(roi_mask_.at<float>(row, (col - 1)) != 0) {
                    // end column index (110)
                    patch_map_[row].push_back(col);
                }
            }
        }
        
        row = row + std::floor(perspective_map_.at<float>(row, 1) * patch_para_ * 2.0f * overlap_ratio_);
    }
#endif
    
#if 0
    // old code for only supporting convex roi region. 
    
    int row = start_row_patch;
    while (row <= end_row_patch) {
        for (int col = 0; col < cols_; col++) {
            if (roi_mask_.at<float>(row, col) != 0) {
                // insert the index of row which contains non-zone elements into vector for
                // obtaining the first non-zero row index and the last non-zero row index.
                patch_map_[row].push_back(col);
                row = row + std::floor(perspective_map_.at<float>(row, 1) * patch_para_ * 2.0f * overlap_ratio_);
                break;
            }
        }
    }

    // handle the situation that extracting patch row is larger than the boundary of extracting
    // patch.
    if (row > end_row_patch) {
        row = end_row_patch;
        // obtain the start column index on the last extracting patch row.
        for (int col = 0; col < cols_; col++) {
            if (roi_mask_.at<float>(row, col) != 0) {
                patch_map_[row].push_back(col);
                break;
            }
        }
    }

    // locate the index of last column for boundary of extracting patches activity.
    for (std::map<int, std::vector<int> >::iterator iter = patch_map_.begin(); iter != patch_map_.end(); iter++) {
        int row = iter->first;
        // obtain the end column index on the last extracting patch row.
        for (int col = patch_map_[row][0]; col < cols_; col++) {
            if (roi_mask_.at<float>(row, col) != 1) {
                patch_map_[row].push_back(col - 1);
                break;
            }
        }
    }
#endif

}

#if 1
// new code for supporting non-convex roi region.
void CDensityExtracter::SetDensityIndex() {    
    density_image_all_index_ = cv::Mat::zeros(rows_ + pad_width_ + pad_width_, cols_ + pad_width_ + pad_width_, CV_32F);
    // Generate patches
    for (std::map<int, std::vector<int> >::iterator iter = patch_map_.begin(); iter != patch_map_.end(); ++iter) {
        int relative_row = iter->first;
        int current_row = relative_row + pad_width_;
        float pers_value = std::floor(perspective_map_.at<float>(relative_row, 1) * patch_para_);
        
        int num_cols_index = iter->second.size();
        if((num_cols_index % 2) != 0) {
            // error condition
            std::cout<<"num_cols_index = "<<num_cols_index<<std::endl;
        }
        
        for (int i = 0; i < num_cols_index; i+=2) {
            int start_col = iter->second[i + 0];
            int end_col = iter->second[i + 1];
            
            int num_cols = std::ceil((end_col - start_col) / (pers_value * 2.0f * overlap_ratio_));
            for (int col_index = 0; col_index < num_cols; ++col_index) {
                int current_col = start_col + col_index * 2.0f * pers_value * overlap_ratio_ + pad_width_;
                patches_.push_back(std::make_pair(cv::Point2f(current_col, current_row), pers_value));
                // Fill density_image_all_index
                for (int row = current_row - pers_value; row < current_row + pers_value+1; ++row) {
                    float* density_index = density_image_all_index_.ptr<float>(row);
                    for (int col = current_col - pers_value; col < current_col + pers_value+1; ++col)
                        density_index[col] += 1;
                }
            }
        }
    }
    cv::max(density_image_all_index_, 1);
    density_image_all_index_ = 1 / density_image_all_index_;
    // Moved from memory_data_laery.cpp, so it will just be called once.
    CHECK_GT(patches_.size(), 0) << "There is no mat to add";
}
#endif

#if 0
// old code for only supporting convex roi region. 

void CDensityExtracter::SetDensityIndex() {
    density_image_all_index_ = cv::Mat::zeros(cols_ + pad_width_ + pad_width_, rows_ + pad_width_ + pad_width_, CV_32F);
    // Generate patches
    for (std::map<int, std::vector<int> >::iterator iter = patch_map_.begin(); iter != patch_map_.end(); ++iter) {
        int relative_row = iter->first;
        int current_row = relative_row + pad_width_;
        float pers_value = std::floor(perspective_map_.at<float>(relative_row, 1) * patch_para_);
        int start_col = iter->second[0];
        int end_col = iter->second[1];
        int num_cols = std::ceil((end_col - start_col) / (pers_value * 2.0f * overlap_ratio_));
        for (int col_index = 0; col_index < num_cols; ++col_index) {
            int current_col = start_col + col_index * 2.0f * pers_value * overlap_ratio_ + pad_width_;
            patches_.push_back(std::make_pair(cv::Point2f(current_col, current_row), pers_value));
            // Fill density_image_all_index
            for (int row = current_row - pers_value; row < current_row + pers_value+1; ++row) {
                float* density_index = density_image_all_index_.ptr<float>(row);
                for (int col = current_col - pers_value; col < current_col + pers_value+1; ++col)
                    density_index[col] += 1;
            }
        }
    }
    cv::max(density_image_all_index_, 1);
    density_image_all_index_ = 1 / density_image_all_index_;
    // Moved from memory_data_laery.cpp, so it will just be called once.
    CHECK_GT(patches_.size(), 0) << "There is no mat to add";
}
#endif

}  // namespace itf
