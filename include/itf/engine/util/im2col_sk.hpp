#ifndef ITF_UTIL_IM2COL_SK_HPP_
#define ITF_UTIL_IM2COL_SK_HPP_

namespace itf {

template <typename Dtype>
void im2col_sk_gpu(const Dtype* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int kstride_h, const int kstride_w, Dtype* data_col);

template <typename Dtype>
void col2im_sk_gpu(const Dtype* data_col, const int channels,
    const int height, const int width, const int patch_h, const int patch_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int kstride_h, const int kstride_w,
    Dtype* data_im);

}  // namespace itf

#endif  // ITF_UTIL_IM2COL_SK_HPP_
