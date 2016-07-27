#ifndef BUFFGPU_H
#define BUFFGPU_H
#include "itf/trackers/trackertype.h"
#include <driver_types.h>
#include <vector_types.h>
#include <vector>

#define TRK_BUFF_LEN 100

template <typename ELEM_T> class BuffGPU
{
public:
    int NQue=0,buff_len=0,tailidx=0;
    ELEM_T *headptr, *tailptr, *d_cur_frame_ptr;
    int* d_lenVec;
    ELEM_T* d_data,*d_curVec;
    BuffGPU();
    ~BuffGPU();
    void init(int n,int l,int fbsize);
    void clear(int* idxVec);
    void updateAFrame(ELEM_T* new_frame_ptr);
    void syncMem();
    inline int trackLen(int i);
    inline ELEM_T* getPtr(int i,int t);
    int* h_lenVec;
    ELEM_T *h_data,*h_curVec,*h_cur_frame_ptr;
};
template <typename ELEM_T> class MemBuff //all syncronos Memory
{
public:
    int byte_size,count_size,elem_size,channel;
    ELEM_T* d_data,*h_data;
    MemBuff(int n,int c=1);
    void SyncD2H();
    void SyncD2HStream(cudaStream_t& stream);
    void SyncH2D();
    void SyncH2DStream(cudaStream_t& stream);
    void updateGPU(ELEM_T* ptr);
    void updateCPU(ELEM_T* ptr);
    void toZeroD();
    void toZeroH();
    void copyFrom(MemBuff<ELEM_T> *src);
    inline ELEM_T* gpu_ptr(){return d_data;}
    inline ELEM_T* cpu_ptr(){return h_data;}
    inline ELEM_T& operator[](int idx){return cpu_ptr()[idx*channel];}
    ELEM_T* gpuAt(int idx){return gpu_ptr()+idx*channel;}
    ELEM_T* cpuAt(int idx){return cpu_ptr()+idx*channel;}
};
template <typename ELEM_T> class BuffInfo //Que Buff Accessor
{
public:
    ELEM_T* data_ptr;
    ELEM_T* cur_ptr;
    ELEM_T* next_ptr;
    int* lenVec;
    int nQue,buffLen,tailidx,channel;
    __host__ __device__ __forceinline__ ELEM_T* getPtr(int i,int t)
    {
        int offset = (tailidx+buffLen-t-1)%buffLen;
        return data_ptr+nQue*offset+i;
    }
    __host__ __device__ __forceinline__ ELEM_T* getPtr(int i){return cur_ptr+i;}
    __host__ __device__ __forceinline__ ELEM_T* getVec(int t)
    {
        int offset = (tailidx+buffLen-t-1)%buffLen;
        return data_ptr+nQue*offset;
    }
};

template class BuffInfo<FeatPts>;

struct TracksInfo
{
    int* lenVec;
    int nQue,buffLen,tailidx;
    FeatPts* trkDataPtr;
    FeatPts* curTrkptr;
    FeatPts* nextTrkptr;
    FeatPts* preTrkptr;

    float* distDataPtr;
    float* curDistPtr;
    float* nextDistPtr;
    float* preDistPtr;

    float2* veloDataPtr;
    float2* curVeloPtr;
    float2* nextVeloPtr;
    float2* preVeloPtr;

    float* spdDataPtr;
    float* curSpdPtr;
    float* nextSpdPtr;
    float* preSpdPtr;

    float* curveDataPtr;
    float2* cumVeloPtr;

    void increPtr(){
        curTrkptr=trkDataPtr+tailidx*nQue;
        curDistPtr=distDataPtr+tailidx*nQue;
        curVeloPtr=veloDataPtr+tailidx*nQue;
        curSpdPtr=spdDataPtr+tailidx*nQue;
        tailidx=(tailidx+1)%buffLen;
        nextTrkptr=trkDataPtr+tailidx*nQue;
        nextDistPtr=distDataPtr+tailidx*nQue;
        nextVeloPtr=veloDataPtr+tailidx*nQue;
        nextSpdPtr=spdDataPtr+tailidx*nQue;
    }
    void init(int n,int l)
    {
        nQue=n,buffLen=l,tailidx=0;
        curTrkptr=trkDataPtr+tailidx*nQue;
        curDistPtr=distDataPtr+tailidx*nQue;
        curVeloPtr=veloDataPtr+tailidx*nQue;
        curSpdPtr=spdDataPtr+tailidx*nQue;
        nextTrkptr=trkDataPtr+tailidx*nQue;
        nextDistPtr=distDataPtr+tailidx*nQue;
        nextVeloPtr=veloDataPtr+tailidx*nQue;
        nextSpdPtr=spdDataPtr+tailidx*nQue;
    }
    template<typename ELEM_T>
    __host__ __device__ __forceinline__ ELEM_T* getPtr_(ELEM_T* data_ptr,int i,int t)
    {
        int offset = (tailidx+buffLen-t-1)%buffLen;
        return data_ptr+nQue*offset+i;
    }

    template<typename ELEM_T>
    __host__ __device__ __forceinline__ ELEM_T* getVec_(ELEM_T* data_ptr,int t)
    {
        int offset = (tailidx+buffLen-t-1)%buffLen;
        return data_ptr+nQue*offset;
    }
};

class Tracks :public TracksInfo//Track Buff
{
public:
    MemBuff<int>* lenData;
    MemBuff<FeatPts>* trkData;
    MemBuff<float2>* veloData;
    MemBuff<float>* distData;
    MemBuff<float>* spdData;

    MemBuff<float>* curveData;
    MemBuff<float2>* cumVelo;

    FeatPts* curCpuPtr;

    void init(int n,int l);
    void Sync();
    void increPtr(){
        curCpuPtr=trkData->cpu_ptr()+tailidx*nQue;
        TracksInfo::increPtr();
    }
    inline FeatPts* getPtr(int i,int t)
    {
        return TracksInfo::getPtr_(trkData->cpu_ptr(),i,t);
    }
    inline FeatPts* getPtr(int i){return curCpuPtr+i;}
    inline int getLen(int i){return lenData->cpu_ptr()[i];}
    inline TracksInfo getInfoGPU()
    {
        TracksInfo info=*this;
        return info;
    }
};
template <typename ELEM_T> class VarType
{
    MemBuff<ELEM_T>* ObjPtr;
    size_t size;
};

class Group
{
public:

    Tracks* tracks;
    int trkPtsNum;
    MemBuff<int>* trkPtsIdx;
    MemBuff<int>* ptsNum;
    MemBuff<float2>* com;
    MemBuff<float2>* velo;
    MemBuff<BBox>* bBox;
    MemBuff<float>* area;
    MemBuff<float>* ptsCorr;
    MemBuff<int>* headStats;
    int* trkPtsIdxPtr;
    int* ptsNumPtr;
    float2* comPtr;
    float2* veloPtr;
    BBox* bBoxPtr;
    float* areaPtr;
    float* ptsCorrPtr;
    int* headStatsPtr;
    /*
     *  to add New Features
     * add MemBuff and ptr in header
     * add definition in init()
     * add Sync in in Sync();
     */
    void init(int maxn,Tracks* trks);
    void SyncD2H();
    void SyncH2D();
    void polySyncH2D();
    void trkPtsSyncD2H();
};

class Groups: public Group
{
public:
    int maxNumGroup,numGroups,kltGroupNum;
    /*
    MemBuff<int>* trkPtsIdx;
    MemBuff<int>* ptsNum;
    MemBuff<float2>* trkPts;
    MemBuff<float2>* com;
    MemBuff<float2>* velo;
    MemBuff<int>* bBox;
    int* trkPtsIdxPtr;
    int* ptsNumPtr;
    float2* trkPtsPtr;
    float2* comPtr;
    float2* veloPtr;
    int* bBoxPtr;
    */
    MemBuff<float2>* shape;
    float2* shapePtr;
    void init(int maxn, Tracks *trks);
    template<typename ELEM_T>
    __host__ __device__ __forceinline__ ELEM_T* getPtr_(ELEM_T* data_ptr,int i,int c=1)
    {
        return data_ptr+i*c;
    }
};

class GroupTrack : public Group
{
public:
    GroupTrack():Group(){inited=0;hide=0;}
    int len,buffLen,tailidx,headidx,inited,hide;
    void inline increPtr()
    {
        tailidx=(tailidx+1)%buffLen;
		len++;
    }
    template<typename ELEM_T>
    __host__ __device__ __forceinline__ ELEM_T* getPtr_(ELEM_T* data_ptr,int t,int c=1)
    {
        int offset = (tailidx+buffLen-t-1)%buffLen;
        return data_ptr+offset*c;
    }
    template<typename ELEM_T>
    __host__ __device__ __forceinline__ ELEM_T* getCur_(ELEM_T* data_ptr,int c=1)
    {
        int offset = (tailidx+buffLen-1)%buffLen;
        return data_ptr+offset*c;
    }

    template<typename ELEM_T>
    __host__ __device__ __forceinline__ ELEM_T* getNext_(ELEM_T* data_ptr,int c=1)
    {
        return data_ptr+tailidx*c;
    }
    BBox* getCurBBox();
    float getCurArea();
	float2* getCurCom();
    float updateCount;
    int trkType;

    void clear();
    void init(int maxn,Tracks* trks);
    void updateFrom(Groups* groups,int idx);

};

class GroupTracks
{
public:
    int numGroup,buffLen,maxNumGroup;
    MemBuff<GroupTrack>* groupTracks;
    MemBuff<int>* vacancy;
	MemBuff<int>* lostvec;
    void init(int maxn);
    GroupTrack* getGroupGPU(int idx)
    {
        return groupTracks->gpu_ptr()+idx;
    }
    GroupTrack& operator[](int idx){return groupTracks->cpu_ptr()[idx];}
    GroupTrack* getPtr(int idx){return groupTracks->cpu_ptr()+idx;}
    BBox *getCurBBox(int i);
    float getCurArea(int i);
    void clear(int idx);
	void lost(int idx);
    int addGroup(Groups* groups, int newIdx);
};

__host__ __device__ __forceinline__ bool ptInBox(int x,int y,BBox& bbox)
{
    return (x>=bbox.left&&x<=bbox.right&&y>=bbox.top&&y<=bbox.bottom);
}
#endif // BUFFGPU_H

