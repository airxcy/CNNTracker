#include "itf/trackers/buffgpu.h"
#include "itf/trackers/gpucommon.hpp"

template <typename ELEM_T>
__global__ void updateVec(ELEM_T* tailptr, ELEM_T* new_frame_ptr,int* lenVec,int buff_len)
{
    int idx=threadIdx.x;
    tailptr[idx]=new_frame_ptr[idx];
    int len =lenVec[idx];
    if(len<buff_len)
    {
        lenVec[idx]++;
    }
}


__global__ void updateFloat2Vec(FeatPts_p tailptr, float2* new_frame_ptr,int* lenVec,int buff_len,int fidx)
{
    int idx=threadIdx.x;
    tailptr[idx].x=new_frame_ptr[idx].x;
    tailptr[idx].y=new_frame_ptr[idx].y;
    tailptr[idx].t=fidx;
    int len =lenVec[idx];
    if(len<buff_len)
    {
        lenVec[idx]++;
    }
}
__global__ void clearTrack(int* lenVec,int* clearIdx)
{
    int idx=threadIdx.x;
    if(clearIdx[idx])
        lenVec[idx]=0;
}

template <typename ELEM_T>
BuffGPU<ELEM_T>::BuffGPU()
{

}
template <typename ELEM_T>
BuffGPU<ELEM_T>::~BuffGPU()
{

}
template <typename ELEM_T>
void BuffGPU<ELEM_T>::init(int n,int l,int fbsize)
{
    NQue=n,buff_len=l;
    gpu_zalloc(d_data,NQue*buff_len,sizeof(ELEM_T));
    h_data=(ELEM_T*)zalloc(NQue*buff_len,sizeof(ELEM_T));

    gpu_zalloc(d_curVec,NQue,sizeof(ELEM_T));
    h_curVec=(ELEM_T*)zalloc(NQue,sizeof(ELEM_T));

    gpu_zalloc(d_lenVec,NQue,sizeof(int));
    h_lenVec=(int*)zalloc(NQue,sizeof(int));

    tailidx=0;
    tailptr=d_data+tailidx*NQue;
    d_cur_frame_ptr=tailptr;
    h_cur_frame_ptr=h_data+tailidx*NQue;
}

template <typename ELEM_T>
void BuffGPU<ELEM_T>::updateAFrame(ELEM_T* new_frame_ptr)
{
    updateVec<<<1,NQue>>>(tailptr,new_frame_ptr,d_lenVec,buff_len);
    d_cur_frame_ptr=tailptr;
    h_cur_frame_ptr=h_data+tailidx*NQue;
    tailidx=(tailidx+1)%buff_len;
    tailptr=d_data+tailidx*NQue;
    //cudaMemcpy(tailptr,new_frame_ptr,NQue*sizeof(ELEM_T),cudaMemcpyDeviceToDevice);
}

template <typename ELEM_T>
void BuffGPU<ELEM_T>::clear(int* idxVec)
{
    clearTrack<<<1,NQue>>>(d_lenVec,idxVec);
}

template <typename ELEM_T>
void BuffGPU<ELEM_T>::syncMem()
{
    cudaMemcpy(h_data,d_data,NQue*buff_len*sizeof(ELEM_T),cudaMemcpyDeviceToHost);
    cudaMemcpy(h_lenVec,d_lenVec,NQue*sizeof(int),cudaMemcpyDeviceToHost);
}

template <typename ELEM_T>
int BuffGPU<ELEM_T>::trackLen(int i)
{
    return h_lenVec[i];
}

template <typename ELEM_T>
ELEM_T* BuffGPU<ELEM_T>::getPtr(int i,int t)
{
    return h_data+t*NQue+i;
}

template class BuffGPU<int>;
template class BuffGPU<uchar1>;
template class BuffGPU<float>;
