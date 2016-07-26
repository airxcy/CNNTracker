#include <cstring>

#include "gtest/gtest.h"

#include "itf/common.hpp"
// #include "caffe/syncedmem.hpp"
// #include "caffe/util/math_functions.hpp"

#include "itf/test/test_itf_main.hpp"

namespace itf {

class CommonTest : public ::testing::Test {};

#ifndef CPU_ONLY  // GPU Caffe singleton test.

TEST_F(CommonTest, TestCublasHandlerGPU) {
  int cuda_device_id;
  CUDA_CHECK(cudaGetDevice(&cuda_device_id));
  EXPECT_TRUE(Caffe::cublas_handle());
}

#endif

/*
TEST_F(CommonTest, TestBrewMode) {
  Caffe::set_mode(Caffe::CPU);
  EXPECT_EQ(Caffe::mode(), Caffe::CPU);
  Caffe::set_mode(Caffe::GPU);
  EXPECT_EQ(Caffe::mode(), Caffe::GPU);
}

TEST_F(CommonTest, TestPhase) {
  Caffe::set_phase(Caffe::TRAIN);
  EXPECT_EQ(Caffe::phase(), Caffe::TRAIN);
  Caffe::set_phase(Caffe::TEST);
  EXPECT_EQ(Caffe::phase(), Caffe::TEST);
}
*/

TEST_F(CommonTest, TestRandSeedCPU) {

  cout<<"test"<<endl;

}

#ifndef CPU_ONLY  // GPU Caffe singleton test.

TEST_F(CommonTest, TestRandSeedGPU) {
  SyncedMemory data_a(10 * sizeof(unsigned int));
  SyncedMemory data_b(10 * sizeof(unsigned int));
  Caffe::set_random_seed(1701);
  CURAND_CHECK(curandGenerate(Caffe::curand_generator(),
        static_cast<unsigned int*>(data_a.mutable_gpu_data()), 10));
  Caffe::set_random_seed(1701);
  CURAND_CHECK(curandGenerate(Caffe::curand_generator(),
        static_cast<unsigned int*>(data_b.mutable_gpu_data()), 10));
  for (int i = 0; i < 10; ++i) {
    EXPECT_EQ(((const unsigned int*)(data_a.cpu_data()))[i],
        ((const unsigned int*)(data_b.cpu_data()))[i]);
  }
}

#endif

}  // namespace caffe
