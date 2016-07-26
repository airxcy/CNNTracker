//
//  flowIOOpenCVWrapper.cpp
//  ITF_Inegrated
//
//  Created by Kun Wang on 9/22/2015.
//  Modified from https://github.com/davidstutz/flow-io-opencv
//  Copyright (c) 2015 CUHK. All rights reserved.
//

#include "itf/crossline/flowIOOpenCVWrapper.h"

int ncols = 0;
#define MAXCOLS 60
int colorwheel[MAXCOLS][3];

void setcols(int r, int g, int b, int k)
{
    colorwheel[k][0] = r;
    colorwheel[k][1] = g;
    colorwheel[k][2] = b;
}

void makecolorwheel()
{
    // relative lengths of color transitions:
    // these are chosen based on perceptual similarity
    // (e.g. one can distinguish more shades between red and yellow 
    //  than between yellow and green)
    int RY = 15;
    int YG = 6;
    int GC = 4;
    int CB = 11;
    int BM = 13;
    int MR = 6;
    ncols = RY + YG + GC + CB + BM + MR;
    //printf("ncols = %d\n", ncols);
    if (ncols > MAXCOLS)
    exit(1);
    int i;
    int k = 0;
    for (i = 0; i < RY; i++) setcols(255,      255*i/RY,     0,        k++);
    for (i = 0; i < YG; i++) setcols(255-255*i/YG, 255,      0,        k++);
    for (i = 0; i < GC; i++) setcols(0,        255,      255*i/GC,     k++);
    for (i = 0; i < CB; i++) setcols(0,        255-255*i/CB, 255,          k++);
    for (i = 0; i < BM; i++) setcols(255*i/BM,     0,        255,          k++);
    for (i = 0; i < MR; i++) setcols(255,      0,        255-255*i/MR, k++);
}

void computeColor(float fx, float fy, unsigned char *pix)
{
    if (ncols == 0)
    makecolorwheel();

    float rad = sqrt(fx * fx + fy * fy);
    float a = atan2(-fy, -fx) / M_PI;
    float fk = (a + 1.0) / 2.0 * (ncols-1);
    int k0 = (int)fk;
    int k1 = (k0 + 1) % ncols;
    float f = fk - k0;
    //f = 0; // uncomment to see original color wheel
    for (int b = 0; b < 3; b++) {
    float col0 = colorwheel[k0][b] / 255.0;
    float col1 = colorwheel[k1][b] / 255.0;
    float col = (1 - f) * col0 + f * col1;
    if (rad <= 1)
        col = 1 - rad * (1 - col); // increase saturation with radius
    else
        col *= .75; // out of range
    pix[2 - b] = (int)(255.0 * col);
    }
}

// return whether flow vector is unknown
bool unknown_flow(float u, float v) {
    return (fabs(u) >  UNKNOWN_FLOW_THRESH) 
    || (fabs(v) >  UNKNOWN_FLOW_THRESH)
    || isnan(u) || isnan(v);
}

bool unknown_flow(float *f) {
    return unknown_flow(f[0], f[1]);
}

void MotionToColor(CFloatImage motim, CByteImage &colim, float maxmotion)
{
    CShape sh = motim.Shape();
    int width = sh.width, height = sh.height;
    colim.ReAllocate(CShape(width, height, 3));
    int x, y;
    // determine motion range:
    float maxx = -999, maxy = -999;
    float minx =  999, miny =  999;
    float maxrad = -1;
    for (y = 0; y < height; y++) {
    for (x = 0; x < width; x++) {
        float fx = motim.Pixel(x, y, 0);
        float fy = motim.Pixel(x, y, 1);
        if (unknown_flow(fx, fy))
        continue;
        maxx = __max(maxx, fx);
        maxy = __max(maxy, fy);
        minx = __min(minx, fx);
        miny = __min(miny, fy);
        float rad = std::sqrt(fx * fx + fy * fy);
        maxrad = __max(maxrad, rad);
    }
    }

    if (maxmotion > 0) // i.e., specified on commandline
    maxrad = maxmotion;

    if (maxrad == 0) // if flow == 0 everywhere
    maxrad = 1;

    for (y = 0; y < height; y++) {
    for (x = 0; x < width; x++) {
        float fx = motim.Pixel(x, y, 0);
        float fy = motim.Pixel(x, y, 1);
        uchar *pix = &colim.Pixel(x, y, 0);
        if (unknown_flow(fx, fy)) {
        pix[0] = pix[1] = pix[2] = 0;
        } else {
        computeColor(fx/maxrad, fy/maxrad, pix);
        }
    }
    }
}

cv::Mat flowToColor(const cv::Mat & flow, float max) {
    assert(flow.channels() == 2);
    
    int rows = flow.rows;
    int cols = flow.cols;
    
    CFloatImage cFlow(cols, rows, 2);
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            cFlow.Pixel(j, i, 0) = flow.at<cv::Vec2f>(i, j)[0];
            cFlow.Pixel(j, i, 1) = flow.at<cv::Vec2f>(i, j)[1];
        }
    }
    
    CByteImage cImage;
    MotionToColor(cFlow, cImage, max);
    
    assert(cImage.Shape().height == rows);
    assert(cImage.Shape().width == cols);
    assert(cImage.Shape().nBands == 3);
    
    cv::Mat image(rows, cols, CV_8UC3, cv::Scalar(0, 0, 0));
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            image.at<cv::Vec3b>(i, j)[0] = cImage.Pixel(j, i, 0);
            image.at<cv::Vec3b>(i, j)[1] = cImage.Pixel(j, i, 1);
            image.at<cv::Vec3b>(i, j)[2] = cImage.Pixel(j, i, 2);
        }
    }    
    return image;
}
