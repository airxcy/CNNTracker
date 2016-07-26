#ifndef ITF_SK_LAYERS_
#define ITF_SK_LAYERS_

#include <string>
#include <utility>
#include <vector>

#include "itf/engine/blob.hpp"
#include "itf/engine/common.hpp"
#include "itf/engine/common_layers.hpp"
#include "itf/engine/data_layers.hpp"
#include "itf/engine/layer.hpp"
#include "itf/engine/loss_layers.hpp"
#include "itf/engine/neuron_layers.hpp"
#include "itf/proto/itf.pb.h"

using std::pair;

namespace itf {

template <typename Dtype>
class ShiftStichLayer : public Layer<Dtype> {
 public:
  explicit ShiftStichLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ShiftStich"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int num_shift_;
  bool shift_or_stich_;
};


template <typename Dtype>
class ConvolutionSKLayer : public Layer<Dtype> {
 public:
  explicit ConvolutionSKLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}

  virtual inline const char* type() const { return "ConvolutionSK"; }
  virtual inline DiagonalAffineMap<Dtype> coord_map() {
    return FilterMap<Dtype>(this->ext_kernel_h_, this->ext_kernel_w_, this->stride_h_,
        this->stride_w_, this->pad_h_, this->pad_w_).inv();
  }

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline bool EqualNumBottomTopBlobs() const { return true; }

  protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int kernel_h_, kernel_w_;
  int ext_kernel_h_, ext_kernel_w_;
  int stride_h_, stride_w_;
  int kstride_h_, kstride_w_;
  int num_;
  int channels_;
  int pad_h_, pad_w_;
  int height_, width_;
  int height_out_, width_out_;
  int num_output_;
  int group_;
  Blob<Dtype> col_buffer_;
  Blob<Dtype> bias_multiplier_;
  bool bias_term_;
  int M_;
  int K_;
  int N_;
};

template <typename Dtype>
class PoolingSKLayer : public Layer<Dtype> {
 public:
  explicit PoolingSKLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Pooling"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int MinTopBlobs() const { return 1; }
  // MAX POOL layers can output an extra top blob for the mask;
  // others can only output the pooled inputs.
  virtual inline int MaxTopBlobs() const {
    return (this->layer_param_.pooling_param().pool() ==
            PoolingParameter_PoolMethod_MAX) ? 2 : 1;
  }
  virtual inline DiagonalAffineMap<Dtype> coord_map() {
    return FilterMap<Dtype>(ext_kernel_h_, ext_kernel_w_, stride_h_, stride_w_,
        pad_h_, pad_w_).inv();
  }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int kernel_h_, kernel_w_;
  int stride_h_, stride_w_;
  int kstride_h_, kstride_w_;
  int ext_kernel_h_, ext_kernel_w_;
  int pad_h_, pad_w_;
  int channels_;
  int height_, width_;
  int pooled_height_, pooled_width_;
  bool global_pooling_;
  Blob<Dtype> rand_idx_;
  Blob<int> max_idx_;
};

}  // namespace itf

#endif  // ITF_SK_LAYERS_
