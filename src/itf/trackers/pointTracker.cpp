#include "itf/trackers/trackers.h"

#include <cmath>
#include <fstream>
#include <string>

#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/gpu/stream_accessor.hpp>

#include <cuda_runtime.h>
#include "itf/trackers/gpucommon.hpp"
#include "itf/trackers/utils.h"
#include "opencv2/gpu/device/common.hpp"
using namespace cv;
using namespace cv::gpu;
char DetectTypeString[NUM_DETECT_TYPE][5]
{
    "GOOD",
    "BAD_",
    "DUPL",
    "SPLT",
    "USED",
    "DLAY",
    "_1V1",
    "MERG",
    "FIRM"
};
void setHW(int w,int h);
void genkernel(float* ptr, int w, int h)
{
    float cx = w / 2, cy = h / 2;
    double theta = 0, sigma_x = w/2, sigma_y=h/2;

    double gaussA = cos(theta) * cos(theta) / 2 / (sigma_x*sigma_x) + sin(theta)*sin(theta) / 2 / (sigma_y*sigma_y)
        , gaussB = -sin(2 * theta) / 4 / (sigma_x*sigma_x) + sin(2 * theta) / 4 / (sigma_y*sigma_y)
        , gaussC = sin(theta) *sin(theta) / 2 / (sigma_x*sigma_x) + cos(theta)*cos(theta) / 2 / (sigma_y*sigma_y);
    for (int i = 0; i < h; i++)
    {
        for (int j = 0; j < w; j++)
        {
            float x = j - cx;
            float y = i - cy;
            float power = -(gaussA * x*x + 2 * gaussB * x*y + gaussC * y*y)*6;
            float val = exp(power);
            ptr[i*w + j] = val;
        }
    }
}

CrowdTracker::CrowdTracker()
{
    frame_width=0, frame_height=0;
    frameidx=0;
    nFeatures=0,nSearch=0; 
    /**cuda **/
    persDone=false;
}
CrowdTracker::~CrowdTracker()
{
    releaseMemory();
}

int CrowdTracker::init(int w, int h,unsigned char* framedata,int nPoints)
{
    /** Checking Device Properties **/
    int nDevices;
    int maxthread=0;
    cudaGetDeviceCount(&nDevices);
    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        
        printf("Device Number: %d\n", i);
        printf("  Device name: %s\n", prop.name);
        printf("  Memory Clock Rate (KHz): %d\n",prop.memoryClockRate);
        printf("  Memory Bus Width (bits): %d\n",prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth (GB/s): %f\n\n",2.0*prop.memoryClockRate*(prop.memoryBusWidth / 8) / 1.0e6);
        std::cout << "maxgridDim" << prop.maxGridSize[0] << "," << prop.maxGridSize[1] << "," << prop.maxGridSize[2] << std::endl;
		std::cout << "maxThreadsPerBlock:" << prop.maxThreadsPerBlock << std::endl;
        

        //cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize,MyKernel, 0, arrayCount);
        if(maxthread==0)maxthread=prop.maxThreadsPerBlock;
        //debuggingFile << prop.major << "," << prop.minor << std::endl;
    }
    //cudaSetDevice(1);
    //std::cout <<"device Status:"<< cudaSetDevice(1) << std::endl;
    /** Basic **/
    frame_width = w,frame_height = h;
	frameSize = frame_width*frame_height;
	frameSizeRGB = frame_width*frame_height*3;
	tailidx = 0, buffLen = 10, mididx = 0, preidx = 0, nextidx = 0;
    setHW(w,h);
    frameidx=0;

	volumeRGB = new MemBuff<unsigned char>(frameSizeRGB*buffLen);
	volumeGray = new MemBuff<unsigned char>(frameSize*buffLen);
    fidxBuff = new MemBuff<int>(buffLen);
	gpu_zalloc(d_rgbframedata, frameSizeRGB, sizeof(unsigned char));
	rgbMat = gpu::GpuMat(frame_height, frame_width, CV_8UC3, d_rgbframedata);
	gpuPreRGBA = gpu::GpuMat(frame_height, frame_width, CV_8UC4);
	gpuRGBA = gpu::GpuMat(frame_height, frame_width, CV_8UC4);

    persMap =  new MemBuff<float>(frame_height*frame_width);
    gpuPersMap= gpu::GpuMat(frame_height, frame_width, CV_32F ,persMap->gpu_ptr());
    roimask =  new MemBuff<unsigned char>(frame_height*frame_width);
    roiMaskMat = gpu::GpuMat(frame_height, frame_width, CV_8UC1 ,roimask->gpu_ptr());
	debuggingFile.open("trackerDump.txt", std::ofstream::out);

    /** Point Tracking and Detecting **/
	nFeatures = maxthread;//(maxthread>1024)?1024:maxthread;
	nFeatures = (maxthread>nPoints) ? nPoints : maxthread;
	nSearch = nFeatures;
	tracksGPU = new Tracks();
	tracksGPU->init(nFeatures, nFeatures);
	//detector=new  gpu::GoodFeaturesToTrackDetector_GPU(nSearch,1e-30,0,3);
	detector = new TargetFinder(nSearch, 1e-30, 3);
	tracker = new  gpu::PyrLKOpticalFlow();
    tracker->winSize = Size(7, 7);
    tracker->maxLevel = 4;
	tracker->iters = 10;

	corners = new MemBuff<int2>(nSearch);
	cornerBuff = new MemBuff<int2>(frameSize);
	totalGradPoint = 0, acceptPtrNum = 0;
	cornerCVStream = gpu::Stream();
	cornerStream = gpu::StreamAccessor::getStream(cornerCVStream);
	detector->_stream = cornerCVStream;
	detector->_stream_t = cornerStream;
	cudaStreamCreate(&CorrStream);

	
	gpuCorners = gpu::GpuMat(1, nSearch, CV_32SC2, corners->gpu_ptr());

	mask = new MemBuff<unsigned char>(frame_height*frame_width);
	maskMat = gpu::GpuMat(frame_height, frame_width, CV_8UC1, mask->gpu_ptr());
	pointRange = new MemBuff<unsigned char>(frame_height*frame_width);
	status = new MemBuff<unsigned char>(nFeatures);
    gpuStatus=gpu::GpuMat(1,nFeatures,CV_8UC1,status->gpu_ptr());
	statusMat = Mat(1, nFeatures, CV_8UC1, status->cpu_ptr());

	
	prePts = Mat(1, nFeatures, CV_32FC2);
	nextPts = Mat(1, nFeatures, CV_32FC2);
	gpuNextPts = GpuMat(1, nFeatures, CV_32FC2);
	gpuPrePts = GpuMat(1, nFeatures, CV_32FC2);


	gradient = new MemBuff<float>(frameSize, CV_32FC1);

    //head Detection
    //initHeadDetector("/home/cyxia/globalCenter/0240_0135/");
    //initHeadDetector("/home/cyxia/headVideos/000001/0240_0135/");
	//Neighbor Search
	nbCount = new MemBuff<int>(nFeatures*nFeatures);
	distCo = new MemBuff<float>(nFeatures*nFeatures);
	cosCo = new MemBuff<float>(nFeatures*nFeatures);
	veloCo = new MemBuff<float>(nFeatures*nFeatures);
	correlation = new MemBuff<float>(nFeatures);

    //k-NN

     //BFsearch
     curK=0,groupN=0;
     bfsearchitems.resize(nFeatures);
     tmpn = (int*)zalloc(nFeatures,sizeof(int));
     idxmap= (int*)zalloc(nFeatures,sizeof(int));
     //label Re-Map
     maxgroupN=0;
     prelabel = new MemBuff<int>(nFeatures);
     label = new MemBuff<int>(nFeatures);
     ptNumInGroup = new MemBuff<int>(nFeatures);

     groups = new Groups();
     groups->init(nFeatures,tracksGPU);
     gType = new MemBuff<uchar>(nFeatures);
     gaussKernel = new MemBuff<float>(KernelH*KernelW);
        genkernel(gaussKernel->cpu_ptr(),KernelW,KernelH);
        gaussKernel->SyncH2D();


     groupsTrk = new GroupTracks();
     maxGroupTrk=180;
     groupsTrk->init(maxGroupTrk);

     streams=std::vector<cudaStream_t>(MAXSTREAM);
     for(int i=0;i<MAXSTREAM;i++)cudaStreamCreate(&streams[i]);
     overLap = new MemBuff<int>(nFeatures*nFeatures);
     matchOld2New = new MemBuff<int>(nFeatures);
     matchNew2Old = new MemBuff<int>(nFeatures);

     rankCountNew = new MemBuff<int>(nFeatures);
     rankCountOld = new MemBuff<int>(nFeatures);
     rankingNew = new MemBuff<int>(nFeatures*nFeatures);
     rankingOld = new MemBuff<int>(nFeatures*nFeatures);
     scoreNew = new MemBuff<float>(nFeatures*nFeatures);
     scoreOld = new MemBuff<float>(nFeatures*nFeatures);
     updateType = new MemBuff<uchar>(nFeatures);
     kltUpdateVec = new MemBuff<float2>(nFeatures);
     kltUpdateBoxVec = new MemBuff<BBoxF>(nFeatures);
     shapevec = new MemBuff<float>(maxGroupTrk);
     mergeTable = std::vector<int>(maxGroupTrk);
     isLinked = std::vector<int>(maxGroupTrk);
     for(int i=0;i<maxGroupTrk;i++)
     {
         gTrkCorr.push_back(std::vector<int>(0));
     }
     gTrkNb = (int *)zalloc(maxGroupTrk*maxGroupTrk,sizeof(int));
     gTrkLabel = (int *)zalloc(maxGroupTrk,sizeof(int));
     gTrkbbNum = (int *)zalloc(maxGroupTrk,sizeof(int));
     gTrkGroupN=0;
     for(int i=0;i<maxGroupTrk;i++)
     {
        mergeIdx.push_back(std::vector<int>(0));
     }
     mergeBox=std::vector<BBox>(maxGroupTrk);
     trkptscount = new MemBuff<int>(maxGroupTrk);
     /**  render **/

     renderMask=new MemBuff<unsigned char>(frame_width*frame_height,3);
     clrvec = new MemBuff<unsigned char>(nFeatures,3);

    /** Self Init **/
    selfinit(framedata);

    debuggingFile<< "inited" << std::endl;
    return 1;
}
void CrowdTracker::initHeadDetector(std::string path)
{

    std::string wstr =path.substr(path.length()-10,4);
    std::string hstr =path.substr(path.length()-5,4);
    std::cout<<"cnn out dim:"<<wstr<<","<<hstr<<std::endl;
    headDetectPath = path;
    headDetectorH=std::stoi(hstr);
    headDetectorW=std::stoi(wstr);
    headDetectorSize=headDetectorH*headDetectorW;
    headScaleW=frame_width/headDetectorW;
    headScaleH=frame_height/headDetectorH;
    headMapOrigin = new MemBuff<float>(headDetectorSize);
    headBinOrigin = new MemBuff<unsigned char>(headDetectorSize);
    headBin = new MemBuff<float>(frameSize);
    headMap = new MemBuff<float>(frameSize);
    headBinUchar = new  MemBuff<uchar>(frameSize);
    connMap = new MemBuff<uchar>(frameSize);
    labelMap = new MemBuff<int>(frameSize);
    mstatsv = new MemBuff<int>(nFeatures,CC_STAT_MAX);
    mcentroidsv = new MemBuff<double2>(nFeatures);
    gHeadCount = new MemBuff<int>(nFeatures);
    gHeadIdxMat = new MemBuff<int>(nFeatures*nFeatures);
}

int CrowdTracker::selfinit(unsigned char* framedata)
{
	Mat curframe(frame_height, frame_width, CV_8UC3, framedata);
	rgbMat.upload(curframe);
	gpuPreGray = GpuMat(frame_height, frame_width, CV_8UC1, volumeGray->gpuAt(mididx*frameSize));
	gpuGray = GpuMat(frame_height, frame_width, CV_8UC1, volumeGray->gpuAt(nextidx*frameSize));
	gpu::cvtColor(rgbMat, gpuGray, CV_RGB2GRAY);


	unsigned char* tmpPtr = volumeGray->gpu_ptr() + tailidx*frameSize;
	cudaMemcpy(tmpPtr, gpuGray.data, frameSize, cudaMemcpyDeviceToDevice);
	tmpPtr = volumeGray->gpu_ptr() + tailidx*frameSizeRGB;
	cudaMemcpy(tmpPtr, rgbMat.data, frameSizeRGB, cudaMemcpyDeviceToDevice);
	tailidx = (tailidx + 1) % buffLen;
	mididx = (tailidx + buffLen / 2) % buffLen;
	preidx = (mididx - 1 + buffLen) % buffLen;
	nextidx = (mididx + 1 + buffLen) % buffLen;
	return true;
}

int CrowdTracker::updateAframe(unsigned char* framedata, int fidx)
{
    std::clock_t start=std::clock();
    curStatus=FINE;
    frameidx=fidx;
    debuggingFile<<"frameidx:"<<frameidx<<std::endl;
    //Mat curframe(frame_height,frame_width,CV_8UC3,framedata);
	cudaMemcpy(d_rgbframedata,framedata, frameSizeRGB, cudaMemcpyHostToDevice);
	unsigned char* tmpPtr = volumeGray->gpu_ptr() + tailidx*frameSize;
	//GpuMat tmpMat(frame_height, frame_width, CV_8UC1, tmpPtr);
	RGB2Gray(d_rgbframedata, tmpPtr);
	//gpu::cvtColor(rgbMat, tmpMat, CV_RGB2GRAY);
	tmpPtr = volumeRGB->gpu_ptr() + tailidx*frameSizeRGB;
	cudaMemcpy(tmpPtr, d_rgbframedata, frameSizeRGB, cudaMemcpyDeviceToDevice);
        (*fidxBuff)[tailidx]=frameidx;
	mididx = (tailidx + buffLen / 2) % buffLen;
	preidx = (mididx - 1 + buffLen) % buffLen;
	nextidx = (mididx + 1 + buffLen) % buffLen;
	tailidx = (tailidx + 1) % buffLen;

	cudaMemcpy(d_rgbframedata, volumeRGB->gpuAt(mididx*frameSizeRGB), frameSizeRGB, cudaMemcpyDeviceToDevice);
	calcGradient();
    detectHead();
	gpuPreGray = GpuMat(frame_height, frame_width, CV_8UC1, volumeGray->gpuAt(preidx*frameSize));
	gpuGray = GpuMat(frame_height, frame_width, CV_8UC1, volumeGray->gpuAt(mididx*frameSize));
	/* rgba KLT tracking
	gpuPreRGB = GpuMat(frame_height, frame_width, CV_8UC3, volumeRGB->gpuAt(preidx*frameSizeRGB));
	gpuRGB = GpuMat(frame_height, frame_width, CV_8UC3, volumeRGB->gpuAt(mididx*frameSizeRGB));
	debuggingFile << "cvtColor"<< std::endl;
	gpu::cvtColor(gpuPreRGB, gpuPreRGBA,CV_RGB2RGBA);
	gpu::cvtColor(gpuRGB, gpuRGBA, CV_RGB2RGBA);
	debuggingFile << "finish cvtColor" << std::endl;
	*/ 

	findPoints();
	PointTracking();
	filterTrackGPU();

    tracksGPU->Sync();
    /** Grouping  **/
    if(groupOnFlag)
    {

        pointCorelate();
        nbCount->SyncD2H();
        cudaMemcpy(bfsearchitems.data(),tracksGPU->lenVec,nFeatures*sizeof(int),cudaMemcpyDeviceToHost);
        for(int i=0;i<nFeatures;i++)
        {
            bfsearchitems[i]=i*(bfsearchitems[i]>0);
        }
        prelabel->copyFrom(label);
        pregroupN = groupN;
        //bfsearch();
        newBFS();
        if(groupN>0&&frameidx>30)
        {
            //updateGroupsTracks();
            if(groupN>maxgroupN)maxgroupN=groupN;


            updateGroupsTracks();
//            unsigned char* h_clrvec=clrvec->cpu_ptr();
//            for(int i=0;i<groupsTrk->numGroup;i++)
//            {
//                HSVtoRGB(h_clrvec+i*3,h_clrvec+i*3+1,h_clrvec+i*3+2,i/(maxgroupN+0.01)*360,1,1);
//            }
            makeGroups();

            clrvec->SyncH2D();
            matchGroups();
            if(groupsTrk->numGroup>1)
            {
                GroupTrkCorrelate();
            }
        }
    }



    Render(framedata);
    PersExcludeMask();
    cudaMemcpy(framedata,d_rgbframedata,frameSizeRGB,cudaMemcpyDeviceToHost);
    float duration = ( std::clock() - start ) / (float) CLOCKS_PER_SEC;
    debuggingFile<<"Total Time"<<duration<<std::endl;
    return 1;
}
void CrowdTracker::updateATrk(int trkIdx,int newIdx)
{
    GroupTrack* gTrk=groupsTrk->getPtr(trkIdx);

    int* gHeadStats = groups->headStats->cpuAt(newIdx);
    float2 updatev;
    int checkSpan = std::min(gTrk->len-1,10);
    int* trkHeadStats =gTrk->getPtr_(gTrk->headStats->cpu_ptr(),checkSpan,CC_STAT_TOTAL);
    if(gTrk->trkType==HEAD_TRK)
    {
//        if(gHeadStats[CC_STAT_AREA]>0)
//        {
//            if (trkHeadStats[CC_STAT_AREA]>0)
//            {
                updatev.x=float(gHeadStats[CC_STAT_HEAD_X]- trkHeadStats[CC_STAT_HEAD_X])/float(checkSpan+1);
                updatev.y=float(gHeadStats[CC_STAT_HEAD_Y]- trkHeadStats[CC_STAT_HEAD_Y])/float(checkSpan+1);
//            }
//            else
//            {
//                updatev=(*groups->velo)[newIdx];
//            }
//        }
//        else
//        {
//            //updatev=*gTrk.getCur_(gTrk.velo->cpu_ptr());
//            updatev=(*groups->velo)[newIdx];
//        }
    }
    else
    {
        updatev=(*groups->velo)[newIdx];
    }

    //updatev.x= updatev.x*0.5+(*kltUpdateVec)[trkIdx].x+0.5;
    //updatev.y= updatev.y*0.5+(*kltUpdateVec)[trkIdx].y+0.5;
    (*kltUpdateVec)[trkIdx].x= updatev.x*0.5+(*kltUpdateVec)[trkIdx].x*0.5;
    (*kltUpdateVec)[trkIdx].y= updatev.y*0.5+(*kltUpdateVec)[trkIdx].y*0.5;
    (*kltUpdateBoxVec)[trkIdx].left=(*groups->bBox)[newIdx].left+updatev.x;
    (*kltUpdateBoxVec)[trkIdx].top=(*groups->bBox)[newIdx].top+updatev.y;
    (*kltUpdateBoxVec)[trkIdx].right=(*groups->bBox)[newIdx].right+updatev.x;
    (*kltUpdateBoxVec)[trkIdx].bottom=(*groups->bBox)[newIdx].bottom+updatev.y;

    gTrk->updateFrom(groups,newIdx);
    memcpy(gTrk->getCur_(gTrk->velo->cpu_ptr()), &updatev, sizeof(float2));
    cudaMemcpy(gTrk->getCur_(gTrk->veloPtr), &updatev, sizeof(float2), cudaMemcpyHostToDevice);

}
void CrowdTracker::updateGroupsTracks()
{
    // Update Tracking Group

    std::clock_t start=std::clock();
    for(int i=0;i<groupsTrk->numGroup;i++)
    {
        if((*groupsTrk->vacancy)[i])
        {
        if(groupsTrk->getPtr(i)->trkType==HEAD_TRK)
        {
            int count = (*rankCountOld)[i];
            if (count == 0)
            {
                //lost
                KLT_updates_Group(i);
                if((*groupsTrk)[i].updateCount>20)
                {
                    groupsTrk->lost(i);
                    (*kltUpdateVec)[i].x=0;
                    (*kltUpdateVec)[i].y=0;
                    (*kltUpdateBoxVec)[i].left=0;
                    (*kltUpdateBoxVec)[i].top=0;
                    (*kltUpdateBoxVec)[i].right=0;
                    (*kltUpdateBoxVec)[i].bottom=0;
                    gTrkCorr[i].clear();
                    isLinked[i]=0;
                }
            }
            else if ((*updateType)[i])
            {
                KLT_updates_Group(i);
                if((*groupsTrk)[i].updateCount>20)
                {
                    groupsTrk->lost(i);
                    (*kltUpdateVec)[i].x=0;
                    (*kltUpdateVec)[i].y=0;
                    (*kltUpdateBoxVec)[i].left=0;
                    (*kltUpdateBoxVec)[i].top=0;
                    (*kltUpdateBoxVec)[i].right=0;
                    (*kltUpdateBoxVec)[i].bottom=0;
                    gTrkCorr[i].clear();
                    isLinked[i]=0;
                }
            }
            else
            {
                int newIdx = (*rankingOld)[i*nFeatures];
                if(newIdx>0)
                {
                    updateATrk(i,newIdx);
                    (*gType)[newIdx]=DETECT_USED;
                }
            }
            if((*groupsTrk)[i].len>30)
            {
                isLinked[i]=1;
            }
        }
        else
        {
            int count = (*rankCountOld)[i];
            if((*updateType)[i]==1)
            {
                KLT_updates_Group(i);
                if((*groupsTrk)[i].updateCount>20)
                {
                    groupsTrk->lost(i);
                    (*kltUpdateVec)[i].x=0;
                    (*kltUpdateVec)[i].y=0;
                    (*kltUpdateBoxVec)[i].left=0;
                    (*kltUpdateBoxVec)[i].top=0;
                    (*kltUpdateBoxVec)[i].right=0;
                    (*kltUpdateBoxVec)[i].bottom=0;
                    gTrkCorr[i].clear();
                    isLinked[i]=0;
                }
            }
            else if (count == 0)
            {
                //lost
                //groupsTrk->lost(i);
                KLT_updates_Group(i);
                if((*groupsTrk)[i].updateCount>10)
                {
                    groupsTrk->lost(i);
                    (*kltUpdateVec)[i].x=0;
                    (*kltUpdateVec)[i].y=0;
                    (*kltUpdateBoxVec)[i].left=0;
                    (*kltUpdateBoxVec)[i].top=0;
                    (*kltUpdateBoxVec)[i].right=0;
                    (*kltUpdateBoxVec)[i].bottom=0;
                    gTrkCorr[i].clear();
                    isLinked[i]=0;
                }
            }
//            else if(count>1)
//            {
//                groupsTrk->lost(i);
//            }
            else if (count >0)
            {
                bool updated=false;
                for(int j=0;j<count;j++)
                {
                    int newIdx = (*rankingOld)[i*nFeatures+j];
                    if(newIdx>0&&(*gType)[newIdx]!=DETECT_USED)
                    {
                        updateATrk(i,newIdx);
                        (*gType)[newIdx]=DETECT_USED;
                        updated=true;
                        break;
                    }
                }
                if(!updated)
                {
                    groupsTrk->lost(i);
                    (*kltUpdateVec)[i].x=0;
                    (*kltUpdateVec)[i].y=0;
                    (*kltUpdateBoxVec)[i].left=0;
                    (*kltUpdateBoxVec)[i].top=0;
                    (*kltUpdateBoxVec)[i].right=0;
                    (*kltUpdateBoxVec)[i].bottom=0;
                    gTrkCorr[i].clear();
                    isLinked[i]=0;
                }
                        //groupsTrk->getPtr(i)->updateFrom(groups, newIdx);
            }
        }
        }
    }
    for(int i=0;i<groupsTrk->numGroup;i++)
    {
        GroupTrack* iTrk=groupsTrk->getPtr(i);
        if(!(*groupsTrk->vacancy)[i])
        {
            for(int j=0;j<maxGroupTrk;j++)
            {
                gTrkNb[i*maxGroupTrk+j]=0;
                gTrkNb[j*maxGroupTrk+i]=0;
            }
        }
    }
    // Adding New Group
    std::cout<<"groupTrk NUm:"<<groupsTrk->numGroup<<std::endl;
    for(int i=1;i<groups->numGroups;i++)
    {
        if ((i>=groups->kltGroupNum&&!(*gType)[i])
                ||(i<groups->kltGroupNum
                   &&(*gType)[i]!=DETECT_USED
                   &&(*gType)[i]!=DETECT_INVALID
                   &&(*gType)[i]!=DETECT_DELAY_MERGE))
        {
            int addidx = groupsTrk->addGroup(groups,i);
            if(addidx>=0)
            {
                if(i<groups->kltGroupNum)
                {
                    groupsTrk->getPtr(addidx)->trkType=KLT_TRK;
                }
                else
                {
                    groupsTrk->getPtr(addidx)->trkType=HEAD_TRK;
                }
                float2 updatev=(*groups->velo)[i];
                (*kltUpdateVec)[addidx]= updatev;
                (*kltUpdateBoxVec)[addidx].left=(*groups->bBox)[i].left+updatev.x;
                (*kltUpdateBoxVec)[addidx].top=(*groups->bBox)[i].top+updatev.y;
                (*kltUpdateBoxVec)[addidx].right=(*groups->bBox)[i].right+updatev.x;
                (*kltUpdateBoxVec)[addidx].bottom=(*groups->bBox)[i].bottom+updatev.y;
            }
            //debuggingFile<<"adding:"<<i<<"added:"<<addidx<<std::endl;
            //BBox* bBox=groupsTrk->getCurBBox(addidx);
            //debuggingFile<<"bBox:"<<bBox->left<<","<<bBox->right<<","<<bBox->top<<","<<bBox->bottom<<std::endl;
        }
    }
    float duration = ( std::clock() - start ) / (float) CLOCKS_PER_SEC;
    debuggingFile<<"Total ReGroup Time"<<duration<<std::endl;
}
void CrowdTracker::newBFS()
{
    label->toZeroH();
    int* h_label=label->cpu_ptr();
    ptNumInGroup->toZeroH();
    int* h_gcount=ptNumInGroup->cpu_ptr();
    int* h_neighbor=nbCount->cpu_ptr();
    memset(idxmap,0,nFeatures*sizeof(int));
    memset(tmpn,0,nFeatures*sizeof(int));

    int idx=0;
    int total=0;
    bool unset=true;
    int gcount=0;
    for(int i=0;i<nFeatures;i++)
    {
        if(unset&&bfsearchitems[i]>0){idx=i;unset=false;}
        total+=(bfsearchitems[i]>0);
        if(!unset)
        {
            tmpn[i]+=(h_neighbor[idx*nFeatures+i]>0);
        }
    }
    bfsearchitems[idx]=0;
    total--;
    debuggingFile<<"total BFS:"<<total<<std::endl;
    curK=1;
    groupN=0;
    gcount++;
    while(total>0)
    {
        int ii=0;
        for(idx=0;idx<nFeatures;idx++)
        {
            if(!ii)ii=idx*(bfsearchitems[idx]>0);
            if(bfsearchitems[idx]&&tmpn[idx])
            {
                int nc=0,nnc=0;
                float nscore=0;
                for(int i=0;i<nFeatures;i++)
                {
                    if(h_neighbor[idx*nFeatures+i])
                    {
                        nc+=(h_neighbor[idx*nFeatures+i]>0);
                        nnc+=(tmpn[i]>0);
                    }
                }
                if(nnc>nc*0.4+1)
                {
                    gcount++;
                    h_label[idx]=curK;
                    for(int i=0;i<nFeatures;i++)
                    {
                        tmpn[i]+=(h_neighbor[idx*nFeatures+i]>0);
                    }
                    bfsearchitems[idx]=0;
                    total--;
                    if(ii==idx)ii=0;
                }
            }
        }
        if(gcount>0)
        {
            h_gcount[curK]+=gcount;
            gcount=0;
        }
        else if(total>0)
        {
            if(h_gcount[curK]>minGSize)
            {
                groupN++;
                idxmap[curK]=groupN;
            }
            curK++;
            gcount=0;
            memset(tmpn,0,nFeatures*sizeof(int));
            idx=ii;
            gcount++;
            h_label[idx]=curK;
            for(int i=0;i<nFeatures;i++)
            {
                tmpn[i]+=(h_neighbor[idx*nFeatures+i]>0);
            }
            bfsearchitems[idx]=0;
            total--;
        }
    }
    for(int i=0;i<nFeatures;i++)
    {
        h_label[i]=idxmap[h_label[i]];
    }
}
void CrowdTracker::PointTracking()
{
	debuggingFile << "tracker" << std::endl;
	tracker->sparse(gpuPreGray, gpuGray, gpuPrePts, gpuNextPts, gpuStatus);
	//tracker->sparse(gpuPreRGBA, gpuRGBA, gpuPrePts, gpuNextPts, gpuStatus);
}
void CrowdTracker::releaseMemory()
{
	tracker->releaseMemory();
	gpuGray.release();
	gpuPreGray.release();
	rgbMat.release();
	gpuPrePts.release();
	gpuNextPts.release();
	gpuStatus.release();
}


void CrowdTracker::setUpPersMap(float* srcMap)
{
	/*
	// camera calibration
	for(int y=0;y<frame_height;y++)
	for(int x=0;x<frame_width;x++)
	{
	float cdist=(frame_width/2.0-abs(x-frame_width/2.0))/frame_width*10;
	srcMap[y*frame_width+x]=srcMap[y*frame_width+x]+cdist*cdist;
	}
	*/
	persMap->updateCPU(srcMap);
	persMap->SyncH2D();
	detector->setPersMat(gpuPersMap, frame_width, frame_height);
	cudaMemcpy(pointRange->gpu_ptr(),detector->rangeMat.data,frameSize,cudaMemcpyHostToDevice);
}
void CrowdTracker::updateROICPU(float* aryPtr, int length)
{
	roimask->toZeroD();
	roimask->toZeroH();
	unsigned char* h_roimask = roimask->cpu_ptr();
	std::vector<Point2f> roivec;
	int counter = 0;
	for (int i = 0; i<length; i++)
	{
		Point2f p(aryPtr[i * 2], aryPtr[i * 2 + 1]);
		roivec.push_back(p);
	}
	for (int i = 0; i<frame_height; i++)
	{
		for (int j = 0; j<frame_width; j++)
		{
			if (pointPolygonTest(roivec, Point2f(j, i), true)>0)
			{
				h_roimask[i*frame_width + j] = 255;
				counter++;

			}
		}
	}

	debuggingFile << counter << std::endl;
	roimask->SyncH2D();
}
void CrowdTracker::updateROImask(unsigned char * ptr)
{
	roimask->toZeroD();
	roimask->toZeroH();
	unsigned char* h_roimask = roimask->cpu_ptr();
	memcpy(h_roimask, ptr, roimask->byte_size);
	roimask->SyncH2D();
}
