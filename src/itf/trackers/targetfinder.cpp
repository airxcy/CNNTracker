

#include "itf/trackers/trackers.h"
#include "itf/trackers/gpucommon.hpp"
//#include "opencv2/core/cuda_devptrs.hpp"
//#include "opencv2/gpu/device/common.hpp"
//#include "opencv2/gpu/device/utility.hpp"
#include "opencv2/gpu/gpu.hpp"

using namespace std;
using namespace cv;
using namespace cv::gpu;
using namespace cv::gpu::device;

int findCorners_gpu(PtrStepSzf eig, float threshold, PtrStepSzb mask, float2* corners, int max_count,cudaStream_t stream);
void sortCorners_gpu(PtrStepSzf eig, float2* corners, int count,cudaStream_t stream);
TargetFinder::TargetFinder(int maxCorners_, double qualityLevel_,
        int blockSize_)
{
    maxCorners = maxCorners_;
    qualityLevel = qualityLevel_;
    blockSize = blockSize_;
    pointBuff.init(2,maxCorners);
}
void TargetFinder::releaseMemory()
{
    Dx_.release();
    Dy_.release();
    buf_.release();
    eig_.release();
    minMaxbuf_.release();
    tmpCorners_.release();
}
void TargetFinder::setPersMat(GpuMat& m,int w,int h)
{
    fw=w,fh=h;
    m.download(cpuPersMap);
    cpuPersMap=cpuPersMap/10-1;
    cpuPersMap.convertTo(rangeMat,CV_8UC1);
    float* tmpptr=(float*)malloc(fw*fh*2*sizeof(float));
    tmpMat=Mat(1, fw*fh, CV_32FC2,tmpptr);
    m.copyTo(persMap);
    gpu::pow(persMap,0.05,persMap);
    double minVal, maxVal;
    gpu::minMax(persMap, & minVal, &maxVal );
    gpu::subtract(persMap,minVal*0.9,persMap);
    //gpu::add(persMap,2,persMap);
}


void TargetFinder::operator ()(const GpuMat& image, GpuMat& corners, const GpuMat& mask)
{

    cornerMinEigenVal(image, eig_, Dx_, Dy_, buf_, blockSize, 3,BORDER_DEFAULT,_stream);
    double maxVal = 0;
    //minMax(eig_, 0, &maxVal, GpuMat(), minMaxbuf_);

    ensureSizeIsEnough(1, std::max(1000, static_cast<int>(image.size().area() * 0.05)), CV_32FC2, tmpCorners_);

    int total = findCorners_gpu(eig_, static_cast<float>(maxVal * qualityLevel), mask, tmpCorners_.ptr<float2>(), tmpCorners_.cols,_stream_t);

    if (total == 0)
    {
        return;
    }

    sortCorners_gpu(eig_, tmpCorners_.ptr<float2>(), total, _stream_t);
    if (rangeMat.empty())
        tmpCorners_.colRange(0, maxCorners > 0 ? std::min(maxCorners, total) : total).copyTo(corners);
    else
    {
        tmpCorners_.colRange(0, total).download(tmpMat);
        float* tmpptr=(float*)tmpMat.data;
        pointBuff.clear();
        float fp2[2];
        for(int i=0;i<total;i++)
        {
            int x =*tmpptr,y=*(tmpptr+1);
            memcpy(fp2,tmpptr,2*sizeof(float));
            tmpptr+=2;
            uchar range=rangeMat.at<uchar>(y,x);
            bool good=true;
            float* ptr2=pointBuff.data;
            for(int j=0;j<pointBuff.len;j++)
            {
                ptr2+=2;
                int x1=*(ptr2),y1=*(ptr2+1);
                int dx=abs(x1-x),dy=abs(y1-y);
                if(dx<range&&dy<range)
                {
                    good=false;
                    break;
                }
            }
            if(good)
            {
                pointBuff.updateAFrame(fp2);
                if(pointBuff.len==maxCorners)
                    break;
            }
        }
        corners.upload(Mat(1, pointBuff.len, CV_32FC2,pointBuff.data));
    }
}
