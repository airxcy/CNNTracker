//
//  flowIOOpenCVWrapper.hpp
//  ITF_Inegrated
//
//  Created by Kun Wang on 9/22/2015.
//  Modified from https://github.com/davidstutz/flow-io-opencv
//  Copyright (c) 2015 CUHK. All rights reserved.
//

#ifndef FLOWIOOPENCVWRAPPER_H_
#define FLOWIOOPENCVWRAPPER_H_

#include <opencv2/highgui/highgui.hpp>
#include "itf/crossline/Image.h"

// the "official" threshold - if the absolute value of either 
// flow component is greater, it's considered unknown
#define UNKNOWN_FLOW_THRESH 1e9
// value to use to represent unknown flow
#define UNKNOWN_FLOW 1e10

// return whether flow vector is unknown
bool unknown_flow(float u, float v);
bool unknown_flow(float *f);

void computeColor(float fx, float fy, unsigned char *pix);
void MotionToColor(CFloatImage motim, CByteImage &colim, float maxmotion);
cv::Mat flowToColor(const cv::Mat & flow, float max = -1);

#endif	// FLOWIOOPENCVWRAPPER_H_
