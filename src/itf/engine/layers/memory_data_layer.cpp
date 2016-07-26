#include <opencv2/core/core.hpp>

#include <vector>

#include "itf/engine/data_layers.hpp"
#include "itf/engine/layer.hpp"
#include "itf/engine/util/io.hpp"

namespace itf {

template <typename Dtype>
void MemoryDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
     const vector<Blob<Dtype>*>& top) {
  batch_size_ = this->layer_param_.memory_data_param().batch_size();
  channels_ = this->layer_param_.memory_data_param().channels();
  height_ = this->layer_param_.memory_data_param().height();
  width_ = this->layer_param_.memory_data_param().width();
  size_ = channels_ * height_ * width_;
  CHECK_GT(batch_size_ * size_, 0) <<
      "batch_size, channels, height, and width must be specified and"
      " positive in memory_data_param";
  vector<int> label_shape(1, batch_size_);
  top[0]->Reshape(batch_size_, channels_, height_, width_);
  top[1]->Reshape(label_shape);
  added_data_.Reshape(batch_size_, channels_, height_, width_);
  added_label_.Reshape(label_shape);
  data_ = NULL;
  labels_ = NULL;
  added_data_.cpu_data();
  added_label_.cpu_data();
}

template <typename Dtype>
void MemoryDataLayer<Dtype>::AddDatumVector(const vector<Datum>& datum_vector) {
  CHECK(!has_new_data_) <<
      "Can't add data until current data has been consumed.";
  size_t num = datum_vector.size();
  CHECK_GT(num, 0) << "There is no datum to add.";
  CHECK_EQ(num % batch_size_, 0) <<
      "The added data must be a multiple of the batch size.";
  added_data_.Reshape(num, channels_, height_, width_);
  added_label_.Reshape(num, 1, 1, 1);
  // Apply data transformations (mirror, scale, crop...)
  this->data_transformer_->Transform(datum_vector, &added_data_);
  // Copy Labels
  Dtype* top_label = added_label_.mutable_cpu_data();
  for (int item_id = 0; item_id < num; ++item_id) {
    top_label[item_id] = datum_vector[item_id].label();
  }
  // num_images == batch_size_
  Dtype* top_data = added_data_.mutable_cpu_data();
  Reset(top_data, top_label, num);
  has_new_data_ = true;
}

// Added by ITF
template <typename Dtype>
void MemoryDataLayer<Dtype>::AddMatVectorDensity(const vector<cv::Mat>& mat_vector) {
    size_t num = mat_vector.size();
    batch_size_ = num;

    CHECK(!has_new_data_) <<
        "Can't add mat until current data has been consumed.";

    added_data_.Reshape(num, channels_, height_, width_);
    added_label_.Reshape(num, 1, 1, 1);
    
    // Apply data transformations (mirror, scale, crop...)
    this->data_transformer_->Transform(mat_vector, &added_data_);
    Dtype* top_label = added_label_.mutable_cpu_data();
    Dtype* top_data = added_data_.mutable_cpu_data();
    Reset(top_data, top_label, num);

    // Skip data transformations
    // Blob<Dtype> uni_blob(1, channels_, height_, width_);
    // for (size_t item_id = 0; item_id < num; ++item_id) {
    //     int offset = added_data_.offset(item_id);
    //     uni_blob.set_cpu_data(added_data_.mutable_cpu_data() + offset);
    //     Dtype* transformed_data = uni_blob.mutable_cpu_data();
    //     int top_index;
    //     for (int h = 0; h < height_; ++h) {
    //       const uchar* ptr = mat_vector[item_id].ptr<uchar>(h);
    //       int img_index = 0;
    //       for (int w = 0; w < width_; ++w) {
    //         for (int c = 0; c < channels_; ++c) {
    //           top_index = (c * height_ + h) * width_ + w;
    //           transformed_data[top_index] = static_cast<Dtype>(ptr[img_index++]);
    //         }
    //       }
    //     }
    // }
    // data_ = added_data_.mutable_cpu_data();
    // labels_ = added_label_.mutable_cpu_data();
    // n_ = num;
    // pos_ = 0;
    
    has_new_data_ = true;
}

template <typename Dtype>
void MemoryDataLayer<Dtype>::AddMatVector(const vector<cv::Mat>& mat_vector, const vector<int>& labels) {
    size_t num = mat_vector.size();
    
    // Added by ITF, begin.
    if (num % batch_size_ != 0) {
        //needs_reshape_ = true;
        batch_size_ = num;
        //added_data_.Reshape(batch_size_, channels_, height_, width_);
        //added_label_.Reshape(batch_size_, 1, 1, 1);
    }
    
    int temp_h = mat_vector[0].rows;
    int temp_w = mat_vector[0].cols;
    
    // Automatically adjust the size according to input images.
    if (height_ != temp_h || width_ != temp_w) {
        //needs_reshape_ = true;
        height_ = temp_h;
        width_ = temp_w;
        size_ = channels_ * height_ * width_;
        //added_data_.Reshape(batch_size_, channels_, height_, width_);
        //added_label_.Reshape(batch_size_, 1, 1, 1);
    }
    // Added by ITF, end.

    CHECK(!has_new_data_) <<
        "Can't add mat until current data has been consumed.";
    CHECK_GT(num, 0) << "There is no mat to add";
    CHECK_EQ(num % batch_size_, 0) <<
        "The added data must be a multiple of the batch size.";
    // zx, Reshape for allocating memory
    added_data_.Reshape(num, channels_, height_, width_);
    added_label_.Reshape(num, 1, 1, 1);
    
    // Apply data transformations (mirror, scale, crop...)
    // zx, Applies the transformation defined in the data layer's transform_param block to a vector of Mat.
    // zx, transform_param block is actually contained in network prototxt. We just consider scale and mean transformation.
    this->data_transformer_->Transform(mat_vector, &added_data_);
    
    // Copy Labels
    // zx, mutable_cpu_data: copy data from gpu(device) to cpu(host)
    Dtype* top_label = added_label_.mutable_cpu_data();
    for (int item_id = 0; item_id < num; ++item_id) {
        top_label[item_id] = labels[item_id];
    }
  
    // num_images == batch_size_
    Dtype* top_data = added_data_.mutable_cpu_data();
    Reset(top_data, top_label, num);
    
    // zx, set a flag to notify the system new data is coming.
    has_new_data_ = true;
}

template <typename Dtype>
void MemoryDataLayer<Dtype>::Reset(Dtype* data, Dtype* labels, int n) {
    CHECK(data);
    CHECK(labels);
    CHECK_EQ(n % batch_size_, 0) << "n must be a multiple of batch size";
    
    // Warn with transformation parameters since a memory array is meant to
    // be generic and no transformations are done with Reset().
    // zx, We comment the following three lines, since we need to transform input data at MemoryDataLayer
    //if (this->layer_param_.has_transform_param()) {
    //    LOG(WARNING) << this->type() << " does not transform array data on Reset()";
    //}
    
    data_ = data;
    labels_ = labels;
    n_ = n;
    pos_ = 0;
}

template <typename Dtype>
void MemoryDataLayer<Dtype>::set_batch_size(int new_size) {
  CHECK(!has_new_data_) <<
      "Can't change batch_size until current data has been consumed.";
  batch_size_ = new_size;
  added_data_.Reshape(batch_size_, channels_, height_, width_);
  added_label_.Reshape(batch_size_, 1, 1, 1);
}

template <typename Dtype>
void MemoryDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK(data_) << "MemoryDataLayer needs to be initalized by calling Reset";

  //if (needs_reshape_) {
  //  top[0]->Reshape(batch_size_, channels_, height_, width_);
  //  top[1]->Reshape(batch_size_, 1, 1, 1);
  //}

  top[0]->Reshape(batch_size_, channels_, height_, width_);
  top[1]->Reshape(batch_size_, 1, 1, 1);
  top[0]->set_cpu_data(data_ + pos_ * size_);
  top[1]->set_cpu_data(labels_ + pos_);
  pos_ = (pos_ + batch_size_) % n_;
  if (pos_ == 0)
    has_new_data_ = false;
}

INSTANTIATE_CLASS(MemoryDataLayer);
REGISTER_LAYER_CLASS(MemoryData);

}  // namespace itf
