//
//  klt_gpu.h
//  ITF_Inegrated
//
//  Created by Chenyang Xia on 8/18/2015.
//  Copyright (c) 2015 CUHK. All rights reserved.
//

#ifndef KLTTRACKER_H
#define KLTTRACKER_H
#include "itf/trackers/buffgpu.h"
#include "itf/trackers/buffers.h"
#include "opencv2/core/core.hpp"
#include "opencv2/gpu/gpu.hpp"

#include <stdint.h>
#include <inttypes.h>
#include <ctime>
#include <fstream>
#include <algorithm>

#define PI 3.14159265

#define minDist 2
#define minGSize 1
#define TIMESPAN 10
#define COSTHRESH 0.4
#define VeloThresh 0.1
#define KnnK 40
#define MoveFactor 0.0001
#define coNBThresh 0
#define minTrkLen 10
#define Pers2Range(pers) pers/6;

#define KernelH 500
#define KernelW 100

typedef int64_t int64;
typedef uint64_t uint64;

namespace cv{

int connectedComponents(cv::InputArray image, cv::OutputArray labels,
                                     int connectivity = 8, int ltype = CV_32S);
int connectedComponentsWithStats(cv::InputArray image, cv::OutputArray labels,
                                              cv::OutputArray stats, cv::OutputArray centroids,
                                              int connectivity = 8, int ltype = CV_32S);
}
class TargetFinder
{
public:
    TargetFinder(int maxCorners = 1000, double qualityLevel = 0.01,
        int blockSize = 3);
    int maxCorners;
    double qualityLevel;
    int blockSize;
    cv::gpu::Stream _stream;
    cudaStream_t _stream_t;
    void releaseMemory();


    void setPersMat(cv::gpu::GpuMat& m,int w,int h);
    //! return 1 rows matrix with CV_32FC2 type
    void operator ()(const cv::gpu::GpuMat& image, cv::gpu::GpuMat& corners, const cv::gpu::GpuMat& mask= cv::gpu::GpuMat());
// private:

    int fw,fh;
    cv::Mat cpuPersMap;
    cv::Mat rangeMat;
    cv::Mat tmpMat;
    Buff<float> pointBuff;
    cv::gpu::GpuMat persMap;
    cv::gpu::GpuMat Dx_;
    cv::gpu::GpuMat Dy_;
    cv::gpu::GpuMat buf_;
    cv::gpu::GpuMat eig_;
    cv::gpu::GpuMat minMaxbuf_;
    cv::gpu::GpuMat tmpCorners_;
};
enum TrackerStatus {FINE=0,TRACKINGERROR};



class CrowdTracker
{
public:
    CrowdTracker();

    ~CrowdTracker();
    /***** CPU *****/
    int init(int w,int h,unsigned char* framedata,int nPoints);
    int selfinit(unsigned char* framedata);
    int updateAframe(unsigned char* framedata,int fidx);
    void releaseMemory();
    void setUpPersMap(float *srcMap);
    void updateROICPU(float* aryPtr,int length);
    void Render(unsigned char * framedata);
	void updateROImask(unsigned char * ptr);
	void RGB2Gray(unsigned char * d_frameRGB, unsigned char* d_frameGray);

    bool isPersDone(){return persDone;}
    void setGroupTrack(bool flag){groupOnFlag=flag;}
    void useRender(bool flag){render=flag;}
    float* getDistCo(){return distCo->cpu_ptr();}
    float* getVeloCo(){return veloCo->cpu_ptr();}
    float* getCosCo(){return cosCo->cpu_ptr();}
    int* getNbCount(){return nbCount->cpu_ptr();}
    Tracks* getTracks(){return tracksGPU;}
    int getNFeatures(){return nFeatures;}
    int getNSearch(){return nSearch;}
    int2* getCorners(){return corners->cpu_ptr();}
    int* getLabel(){return label->cpu_ptr();}
    int* getPreLabel(){return prelabel->cpu_ptr();}
    unsigned char* getClrvec(){return clrvec->cpu_ptr();}
    Groups& getGroups(){return *groups;}
    GroupTracks& getGroupsTrk(){return *groupsTrk;}

    TrackerStatus curStatus;
	std::ofstream debuggingFile;
//private:
    /** Basic **/
    int frame_width=0, frame_height=0;
    int frameidx=0;
    bool persDone=false,render=true;


    /** Point Tracking and Detecting **/
    int nFeatures=0,nSearch=0;
	TracksInfo trkInfo;
	Tracks* tracksGPU;
	

	void PointTracking();
	void findPoints();
	void filterTrackGPU();
	void PersExcludeMask();

    /***** GPU *****/
	cudaStream_t cornerStream;
	cv::gpu::Stream cornerCVStream;
	cudaStream_t CorrStream;
	std::vector<cudaStream_t> streams;

	TargetFinder* detector;
    cv::gpu::PyrLKOpticalFlow* tracker;
	cv::Mat Gray,preGray,prePts, nextPts, statusMat;
	cv::gpu::GpuMat gpuGray, gpuPreGray, gpuRGB, gpuPreRGB , gpuRGBA, gpuPreRGBA, rgbMat, maskMat, roiMaskMat, gpuPersMap, gpuSegMat,
		gpuCorners, gpuPrePts, gpuNextPts, gpuStatus;
	MemBuff<float>* gradient;
	void calcGradient();
	
    //Basic
	unsigned char* d_rgbframedata;
    MemBuff<float>* persMap;
	MemBuff<unsigned char> *mask, *roimask, *segmask, *segNeg;
	int tailidx, frameSize, buffLen, mididx, frameSizeRGB, preidx, nextidx;
	MemBuff<unsigned char> *volumeGray;
	MemBuff<unsigned char> *volumeRGB;
    MemBuff<int> *fidxBuff;
	//Rendering
	MemBuff<unsigned char>* renderMask;
	MemBuff<unsigned char>* clrvec;

	//Tracking
	MemBuff<unsigned char> *status, *pointRange;
	MemBuff<int2>* corners;

	int totalGradPoint, acceptPtrNum;
	MemBuff<int2>* cornerBuff;
    //head Detect
    MemBuff<float>* headMap;
    MemBuff<float>* headMapOrigin;
    MemBuff<uchar>* headBinOrigin;
    MemBuff<float>* headBin;
    MemBuff<uchar>* headBinUchar;
    MemBuff<uchar>* connMap;
    MemBuff<int>* labelMap;
    MemBuff<int>* mstatsv;
    MemBuff<double2>* mcentroidsv;
    MemBuff<int>* gHeadCount;
    MemBuff<int>* gHeadIdxMat;

    int nHeadLabel;
    float headScaleW,headScaleH;
    int headDetectorW,headDetectorH,headDetectorSize;
    double headMax,headMin;
    std::string headDetectPath;
    std::ifstream headFileReader;
    MemBuff<int2>* headPts;




    char strbuff[100];
    void initHeadDetector(std::string path);
    void detectHead();


    //Correlate
	void pointCorelate();
	MemBuff<int>* nbCount;
	MemBuff<float>* distCo, *cosCo, *veloCo, *correlation;

    /** Grouping **/
    bool groupOnFlag=true;
    int curK=0,pregroupN=0,groupN=0,maxgroupN=0;
    int totalEdge;

    std::vector<int> bfsearchitems;
    void bfsearch();
    void newBFS();
    void makeGroups();
    void matchScorePixel();
    void matchScoreSimple();
    void matchGroups();
    void splitGroup();
    void addHeadBox();
    void rankDetection();
    void renderPts();
    void updateGroupsTracks();
    void updateATrk(int trkIdx,int newIdx);
    void KLT_updates_Group(int );
    MemBuff<int>* prelabel,* label,* ptNumInGroup;


    int *tmpn,*idxmap;
    Groups* groups;
    GroupTracks* groupsTrk;
    MemBuff<int>* overLap;
    MemBuff<int>* matchOld2New,* matchNew2Old;
    MemBuff<int>* rankingOld,* rankingNew,* rankCountOld,* rankCountNew;
    MemBuff<float>* scoreOld,* scoreNew;
    MemBuff<uchar>* updateType;
    MemBuff<float2>* kltUpdateVec;
    MemBuff<BBoxF>* kltUpdateBoxVec;

    std::vector< int > mergeTable;
    std::vector<BBox> mergeBox;
    std::vector<std::vector<int> > mergeIdx;
    int* gTrkNb;
    int* gTrkLabel;
    int* gTrkbbNum;
    int gTrkGroupN;

    std::vector< int> isLinked;
    MemBuff<float>* shapevec;
    int maxGroupTrk;
    MemBuff<float>* gCosCo;
    MemBuff<float>* gVeloCo;
    MemBuff<float>* gDistCo;
    MemBuff<int>* gNbCount;
    MemBuff<int>* trkptscount;
    void GroupTrkCorrelate();

    std::vector< std::vector<int> > gTrkCorr;
    MemBuff<uchar>* gType;
    MemBuff<float>* gaussKernel;

    //cublasHandle_t handle;
};

#endif
