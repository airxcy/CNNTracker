#include "itf/trackers/trackers.h"
#include "itf/trackers/gpucommon.hpp"
#include "itf/trackers/utils.h"
#include "thrust/sort.h"
#include <iostream>
#include <stdio.h>
#include <numeric>
#include <opencv2/gpu/device/common.hpp>
//#include <cudnn.h>

using namespace cv;
//using namespace cv::gpu;
__device__ int d_framewidth[1],d_frameheight[1];
__device__ int d_buffLen[1], d_tailidx[1],d_total[1],d_maxVal[1];
__device__ int lockOld[NUMTHREAD],lockNew[NUMTHREAD];
__device__ int sobelFilter[3 * 3 * 3];
__device__ unsigned char  x3x3[27], y3x3[27], z3x3[27];
__device__ unsigned char d_clrvec[3*1000];


void __global__ clearLockKernel()
{
    lockOld[threadIdx.x]=0;
    lockNew[threadIdx.x]=0;
}
void clearLock()
{
    clearLockKernel<<<1,NUMTHREAD>>>();
}


void setHW(int w, int h)
{
	cudaMemcpyToSymbol(d_framewidth, &w, sizeof(int));
	cudaMemcpyToSymbol(d_frameheight, &h, sizeof(int));
	int tmpsobel[3 * 3 * 3] =
	{
		1, 2, 1, 2, 4, 2, 1, 2, 1,
		0, 0, 0, 0, 0, 0, 0, 0, 0,
	   -1,-2,-1,-2,-4,-2,-1,-2,-1
	};
	cudaMemcpyToSymbol(sobelFilter, tmpsobel, sizeof(int) * 27);
	unsigned char tmpz[27] =
	{
		0, 0, 0, 0, 0, 0, 0, 0, 0,
		1, 1, 1, 1, 1, 1, 1, 1, 1,
		2, 2, 2, 2, 2, 2, 2, 2, 2
	};
	cudaMemcpyToSymbol(z3x3, tmpz, sizeof(unsigned char) * 27);
	unsigned char tmpy[27] =
	{
		0, 0, 0, 1, 1, 1, 2, 2, 2,
		0, 0, 0, 1, 1, 1, 2, 2, 2,
		0, 0, 0, 1, 1, 1, 2, 2, 2
	};
	cudaMemcpyToSymbol(y3x3, tmpy, sizeof(unsigned char) * 27);
	unsigned char tmpx[27] =
	{
		0, 1, 2, 0, 1, 2, 0, 1, 2,
		0, 1, 2, 0, 1, 2, 0, 1, 2,
		0, 1, 2, 0, 1, 2, 0, 1, 2
	};
	cudaMemcpyToSymbol(x3x3, tmpx, sizeof(unsigned char) * 27);
	unsigned char clrve[1000 * 3];
	for(int i=0;i<360;i++)
	{
		HSVtoRGB(clrve + i * 3, clrve + i * 3 + 1, clrve + i * 3 + 2, i+120.0, 1, 1);
	}
	cudaMemcpyToSymbol(d_clrvec, clrve, sizeof(unsigned char) * 3000);
}
texture <unsigned char, cudaTextureType2D, cudaReadModeElementType> volumeTexture(0, cudaFilterModePoint, cudaAddressModeClamp);

#define applyKernel3x3(x,y,z) \
result += sobelFilter[(z[0] * 3 + y[0]) * 3 + x[0]] * volume[(zs[0] * fh + ys[0])*fw + xs[0]];\
result += sobelFilter[(z[1] * 3 + y[1]) * 3 + x[1]] * volume[(zs[0] * fh + ys[0])*fw + xs[1]];\
result += sobelFilter[(z[2] * 3 + y[2]) * 3 + x[2]] * volume[(zs[0] * fh + ys[0])*fw + xs[2]];\
result += sobelFilter[(z[3] * 3 + y[3]) * 3 + x[3]] * volume[(zs[0] * fh + ys[1])*fw + xs[0]];\
result += sobelFilter[(z[4] * 3 + y[4]) * 3 + x[4]] * volume[(zs[0] * fh + ys[1])*fw + xs[1]];\
result += sobelFilter[(z[5] * 3 + y[5]) * 3 + x[5]] * volume[(zs[0] * fh + ys[1])*fw + xs[2]];\
result += sobelFilter[(z[6] * 3 + y[6]) * 3 + x[6]] * volume[(zs[0] * fh + ys[2])*fw + xs[0]];\
result += sobelFilter[(z[7] * 3 + y[7]) * 3 + x[7]] * volume[(zs[0] * fh + ys[2])*fw + xs[1]];\
result += sobelFilter[(z[8] * 3 + y[8]) * 3 + x[8]] * volume[(zs[0] * fh + ys[2])*fw + xs[2]];\
	\
result += sobelFilter[(z[9] * 3 + y[9]) * 3 + x[9]] * volume[(zs[1] * fh + ys[0])*fw + xs[0]];\
result += sobelFilter[(z[10] * 3 + y[10]) * 3 + x[10]] * volume[(zs[1] * fh + ys[0])*fw + xs[1]];\
result += sobelFilter[(z[11] * 3 + y[11]) * 3 + x[11]] * volume[(zs[1] * fh + ys[0])*fw + xs[2]];\
result += sobelFilter[(z[12] * 3 + y[12]) * 3 + x[12]] * volume[(zs[1] * fh + ys[1])*fw + xs[0]];\
result += sobelFilter[(z[13] * 3 + y[13]) * 3 + x[13]] * volume[(zs[1] * fh + ys[1])*fw + xs[1]];\
result += sobelFilter[(z[14] * 3 + y[14]) * 3 + x[14]] * volume[(zs[1] * fh + ys[1])*fw + xs[2]];\
result += sobelFilter[(z[15] * 3 + y[15]) * 3 + x[15]] * volume[(zs[1] * fh + ys[2])*fw + xs[0]];\
result += sobelFilter[(z[16] * 3 + y[16]) * 3 + x[16]] * volume[(zs[1] * fh + ys[2])*fw + xs[1]];\
result += sobelFilter[(z[17] * 3 + y[17]) * 3 + x[17]] * volume[(zs[1] * fh + ys[2])*fw + xs[2]];\
	\
result += sobelFilter[(z[18] * 3 + y[18]) * 3 + x[18]] * volume[(zs[2] * fh + ys[0])*fw + xs[0]];\
result += sobelFilter[(z[19] * 3 + y[19]) * 3 + x[19]] * volume[(zs[2] * fh + ys[0])*fw + xs[1]];\
result += sobelFilter[(z[20] * 3 + y[20]) * 3 + x[20]] * volume[(zs[2] * fh + ys[0])*fw + xs[2]];\
result += sobelFilter[(z[21] * 3 + y[21]) * 3 + x[21]] * volume[(zs[2] * fh + ys[1])*fw + xs[0]];\
result += sobelFilter[(z[22] * 3 + y[22]) * 3 + x[22]] * volume[(zs[2] * fh + ys[1])*fw + xs[1]];\
result += sobelFilter[(z[23] * 3 + y[23]) * 3 + x[23]] * volume[(zs[2] * fh + ys[1])*fw + xs[2]];\
result += sobelFilter[(z[24] * 3 + y[24]) * 3 + x[24]] * volume[(zs[2] * fh + ys[2])*fw + xs[0]];\
result += sobelFilter[(z[25] * 3 + y[25]) * 3 + x[25]] * volume[(zs[2] * fh + ys[2])*fw + xs[1]];\
result += sobelFilter[(z[26] * 3 + y[26]) * 3 + x[26]] * volume[(zs[2] * fh + ys[2])*fw + xs[2]];

__global__ void gradientKernel(unsigned char* volume, unsigned char* rgbframe, unsigned char* rframe, unsigned char* gframe, int z0)
{
	int buffLen = d_buffLen[0];
	int fw = d_framewidth[0];
	int fh = d_frameheight[0];
	
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	
	const int z_1 = (z0 - 1 + buffLen) % buffLen;
	const int z1 = (z0 + 1) % buffLen;
	const int xs[3] = { x - 1, x, x + 1 };
	const int ys[3] = { y - 1, y, y + 1 };
	const int zs[3] = { z_1, z0, z1 };

	//for(int i 
	if (x>1 && y>1 && x < fw - 1 && y < fh - 1)
	{
		int offset = y*fw + x;
		int offset3 = offset * 3;
		//rgbframe[offset3] =  abs(rframe[offset]);
		//rgbframe[offset3 + 1] = abs(gframe[offset ]);
		//rgbframe[offset3 + 2] = 0;
		int result = 0, result1=0;
		float val1 = 0;
		/*
		for(int zi =0;zi<3;zi++)
			for (int yi = 0; yi < 3;yi++)
				for (int xi = 0; xi < 3; xi++)
				{
					result += sobelFilter[(zi * 3 + yi) * 3 + xi] * volume[((z-zi+1)*fh+(y+yi-1))*fw + x+xi-1];
					
				}
		*/
		/*
		result += sobelFilter[(0 * 3 + 0) * 3 + 0] * volume[(zs[0] * fh + ys[0])*fw + xs[0]];
		result += sobelFilter[(0 * 3 + 0) * 3 + 1] * volume[(zs[0] * fh + ys[0])*fw + xs[1]];
		result += sobelFilter[(0 * 3 + 0) * 3 + 2] * volume[(zs[0] * fh + ys[0])*fw + xs[2]];
		result += sobelFilter[(0 * 3 + 1) * 3 + 0] * volume[(zs[0] * fh + ys[1])*fw + xs[0]];
		result += sobelFilter[(0 * 3 + 1) * 3 + 1] * volume[(zs[0] * fh + ys[1])*fw + xs[1]];
		result += sobelFilter[(0 * 3 + 1) * 3 + 2] * volume[(zs[0] * fh + ys[1])*fw + xs[2]];
		result += sobelFilter[(0 * 3 + 2) * 3 + 0] * volume[(zs[0] * fh + ys[2])*fw + xs[0]];
		result += sobelFilter[(0 * 3 + 2) * 3 + 1] * volume[(zs[0] * fh + ys[2])*fw + xs[1]];
		result += sobelFilter[(0 * 3 + 2) * 3 + 2] * volume[(zs[0] * fh + ys[2])*fw + xs[2]];
		
		result += sobelFilter[(1 * 3 + 0) * 3 + 0] * volume[(zs[1] * fh + ys[0])*fw + xs[0]];
		result += sobelFilter[(1 * 3 + 0) * 3 + 1] * volume[(zs[1] * fh + ys[0])*fw + xs[1]];
		result += sobelFilter[(1 * 3 + 0) * 3 + 2] * volume[(zs[1] * fh + ys[0])*fw + xs[2]];
		result += sobelFilter[(1 * 3 + 1) * 3 + 0] * volume[(zs[1] * fh + ys[1])*fw + xs[0]];
		result += sobelFilter[(1 * 3 + 1) * 3 + 1] * volume[(zs[1] * fh + ys[1])*fw + xs[1]];
		result += sobelFilter[(1 * 3 + 1) * 3 + 2] * volume[(zs[1] * fh + ys[1])*fw + xs[2]];
		result += sobelFilter[(1 * 3 + 2) * 3 + 0] * volume[(zs[1] * fh + ys[2])*fw + xs[0]];
		result += sobelFilter[(1 * 3 + 2) * 3 + 1] * volume[(zs[1] * fh + ys[2])*fw + xs[1]];
		result += sobelFilter[(1 * 3 + 2) * 3 + 2] * volume[(zs[1] * fh + ys[2])*fw + xs[2]];

		result += sobelFilter[(2 * 3 + 0) * 3 + 0] * volume[(zs[2] * fh + ys[0])*fw + xs[0]];
		result += sobelFilter[(2 * 3 + 0) * 3 + 1] * volume[(zs[2] * fh + ys[0])*fw + xs[1]];
		result += sobelFilter[(2 * 3 + 0) * 3 + 2] * volume[(zs[2] * fh + ys[0])*fw + xs[2]];
		result += sobelFilter[(2 * 3 + 1) * 3 + 0] * volume[(zs[2] * fh + ys[1])*fw + xs[0]];
		result += sobelFilter[(2 * 3 + 1) * 3 + 1] * volume[(zs[2] * fh + ys[1])*fw + xs[1]];
		result += sobelFilter[(2 * 3 + 1) * 3 + 2] * volume[(zs[2] * fh + ys[1])*fw + xs[2]];
		result += sobelFilter[(2 * 3 + 2) * 3 + 0] * volume[(zs[2] * fh + ys[2])*fw + xs[0]];
		result += sobelFilter[(2 * 3 + 2) * 3 + 1] * volume[(zs[2] * fh + ys[2])*fw + xs[1]];
		result += sobelFilter[(2 * 3 + 2) * 3 + 2] * volume[(zs[2] * fh + ys[2])*fw + xs[2]];


		result1 += sobelFilter[(0 * 3 + 0) * 3 + 0] * volume[(zs[2] * fh + ys[0])*fw + xs[0]];
		result1 += sobelFilter[(0 * 3 + 0) * 3 + 1] * volume[(zs[2] * fh + ys[0])*fw + xs[1]];
		result1 += sobelFilter[(0 * 3 + 0) * 3 + 2] * volume[(zs[2] * fh + ys[0])*fw + xs[2]];
		result1 += sobelFilter[(0 * 3 + 1) * 3 + 0] * volume[(zs[2] * fh + ys[1])*fw + xs[0]];
		result1 += sobelFilter[(0 * 3 + 1) * 3 + 1] * volume[(zs[2] * fh + ys[1])*fw + xs[1]];
		result1 += sobelFilter[(0 * 3 + 1) * 3 + 2] * volume[(zs[2] * fh + ys[1])*fw + xs[2]];
		result1 += sobelFilter[(0 * 3 + 2) * 3 + 0] * volume[(zs[2] * fh + ys[2])*fw + xs[0]];
		result1 += sobelFilter[(0 * 3 + 2) * 3 + 1] * volume[(zs[2] * fh + ys[2])*fw + xs[1]];
		result1 += sobelFilter[(0 * 3 + 2) * 3 + 2] * volume[(zs[2] * fh + ys[2])*fw + xs[2]];

		result1 += sobelFilter[(1 * 3 + 0) * 3 + 0] * volume[(zs[1] * fh + ys[0])*fw + xs[0]];
		result1 += sobelFilter[(1 * 3 + 0) * 3 + 1] * volume[(zs[1] * fh + ys[0])*fw + xs[1]];
		result1 += sobelFilter[(1 * 3 + 0) * 3 + 2] * volume[(zs[1] * fh + ys[0])*fw + xs[2]];
		result1 += sobelFilter[(1 * 3 + 1) * 3 + 0] * volume[(zs[1] * fh + ys[1])*fw + xs[0]];
		result1 += sobelFilter[(1 * 3 + 1) * 3 + 1] * volume[(zs[1] * fh + ys[1])*fw + xs[1]];
		result1 += sobelFilter[(1 * 3 + 1) * 3 + 2] * volume[(zs[1] * fh + ys[1])*fw + xs[2]];
		result1 += sobelFilter[(1 * 3 + 2) * 3 + 0] * volume[(zs[1] * fh + ys[2])*fw + xs[0]];
		result1 += sobelFilter[(1 * 3 + 2) * 3 + 1] * volume[(zs[1] * fh + ys[2])*fw + xs[1]];
		result1 += sobelFilter[(1 * 3 + 2) * 3 + 2] * volume[(zs[1] * fh + ys[2])*fw + xs[2]];

		result1 += sobelFilter[(2 * 3 + 0) * 3 + 0] * volume[(zs[0] * fh + ys[0])*fw + xs[0]];
		result1 += sobelFilter[(2 * 3 + 0) * 3 + 1] * volume[(zs[0] * fh + ys[0])*fw + xs[1]];
		result1 += sobelFilter[(2 * 3 + 0) * 3 + 2] * volume[(zs[0] * fh + ys[0])*fw + xs[2]];
		result1 += sobelFilter[(2 * 3 + 1) * 3 + 0] * volume[(zs[0] * fh + ys[1])*fw + xs[0]];
		result1 += sobelFilter[(2 * 3 + 1) * 3 + 1] * volume[(zs[0] * fh + ys[1])*fw + xs[1]];
		result1 += sobelFilter[(2 * 3 + 1) * 3 + 2] * volume[(zs[0] * fh + ys[1])*fw + xs[2]];
		result1 += sobelFilter[(2 * 3 + 2) * 3 + 0] * volume[(zs[0] * fh + ys[2])*fw + xs[0]];
		result1 += sobelFilter[(2 * 3 + 2) * 3 + 1] * volume[(zs[0] * fh + ys[2])*fw + xs[1]];
		result1 += sobelFilter[(2 * 3 + 2) * 3 + 2] * volume[(zs[0] * fh + ys[2])*fw + xs[2]];
		*/
		float xgrad = 0, ygrad = 0, zgrad = 0;
		applyKernel3x3(x3x3, y3x3, z3x3);
			//rgbframe[offset3] = (abs(result) + abs(result1)) / 16.0;
				zgrad = result / 8.0;
				rgbframe[offset3] = (zgrad);
		result = 0;

		//rgbframe[offset3+1] = volume[offset+fw*fh*z0];
		//rgbframe[offset3+2] = volume[offset + fw*fh*z0];
		
		applyKernel3x3(z3x3, x3x3, y3x3);
			//rgbframe[offset3 + 1] = (abs(result) + abs(result1)) / 16.0;
				ygrad = result / 16.0;
				//rgbframe[offset3 + 1] = abs(ygrad);
		result = 0;
		applyKernel3x3(y3x3, z3x3, x3x3);
			//rgbframe[offset3 + 2] = (abs(result) + abs(result1)) / 16.0;
				xgrad = result / 16.0;
				//rgbframe[offset3 + 2] = abs(xgrad);
		result = 0;
		
		float grad = sqrt(xgrad*xgrad + ygrad*ygrad + zgrad*zgrad);
		//if (x<fw / 2)result = abs(result);

		
		rgbframe[offset3] = grad;
		
		rgbframe[offset3 + 1] = volume[offset + fw*fh*z0];
		rgbframe[offset3 + 2] = volume[offset + fw*fh*z0];
		
	}
}

__global__ void FastSobel_t(unsigned char* volume,unsigned char* volumeRGB, int z0, float* gradient
	,unsigned char* rgb_showFrame)
{
	int buffLen = d_buffLen[0];
	int fw = d_framewidth[0];
	int fh = d_frameheight[0];

	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	const int z_1 = (z0 - 1 + buffLen) % buffLen;
	const int z1 = (z0 + 1) % buffLen;
	const int xs[3] = { x - 1, x, x + 1 };
	const int ys[3] = { y - 1, y, y + 1 };
	const int zs[3] = { z_1, z0, z1 };
	unsigned char* rgbframe = volumeRGB;// + z0*fh*fw*3;
	//for(int i 
	if (x>1 && y>1 && x < fw - 1 && y < fh - 1)
	{
		int offset = y*fw + x;
		int offset3 = offset * 3;
		//rgbframe[offset3] =  abs(rframe[offset]);
		//rgbframe[offset3 + 1] = abs(gframe[offset ]);
		//rgbframe[offset3 + 2] = 0;
		int result = 0, result1 = 0;
		float val1 = 0;
		
		result += sobelFilter[0] * volume[(zs[0] * fh + ys[0])*fw + xs[0]];
		result += sobelFilter[1] * volume[(zs[0] * fh + ys[0])*fw + xs[1]];
		result += sobelFilter[2] * volume[(zs[0] * fh + ys[0])*fw + xs[2]];
		result += sobelFilter[3] * volume[(zs[0] * fh + ys[1])*fw + xs[0]];
		result += sobelFilter[4] * volume[(zs[0] * fh + ys[1])*fw + xs[1]];
		result += sobelFilter[5] * volume[(zs[0] * fh + ys[1])*fw + xs[2]];
		result += sobelFilter[6] * volume[(zs[0] * fh + ys[2])*fw + xs[0]];
		result += sobelFilter[7] * volume[(zs[0] * fh + ys[2])*fw + xs[1]];
		result += sobelFilter[8] * volume[(zs[0] * fh + ys[2])*fw + xs[2]];
		/*
		result += sobelFilter[9] * volume[(zs[1] * fh + ys[0])*fw + xs[0]];
		result += sobelFilter[10] * volume[(zs[1] * fh + ys[0])*fw + xs[1]];
		result += sobelFilter[11] * volume[(zs[1] * fh + ys[0])*fw + xs[2]];
		result += sobelFilter[12] * volume[(zs[1] * fh + ys[1])*fw + xs[0]];
		result += sobelFilter[13] * volume[(zs[1] * fh + ys[1])*fw + xs[1]];
		result += sobelFilter[14] * volume[(zs[1] * fh + ys[1])*fw + xs[2]];
		result += sobelFilter[15] * volume[(zs[1] * fh + ys[2])*fw + xs[0]];
		result += sobelFilter[16] * volume[(zs[1] * fh + ys[2])*fw + xs[1]];
		result += sobelFilter[17] * volume[(zs[1] * fh + ys[2])*fw + xs[2]];
		*/
		result += sobelFilter[18] * volume[(zs[2] * fh + ys[0])*fw + xs[0]];
		result += sobelFilter[19] * volume[(zs[2] * fh + ys[0])*fw + xs[1]];
		result += sobelFilter[20] * volume[(zs[2] * fh + ys[0])*fw + xs[2]];
		result += sobelFilter[21] * volume[(zs[2] * fh + ys[1])*fw + xs[0]];
		result += sobelFilter[22] * volume[(zs[2] * fh + ys[1])*fw + xs[1]];
		result += sobelFilter[23] * volume[(zs[2] * fh + ys[1])*fw + xs[2]];
		result += sobelFilter[24] * volume[(zs[2] * fh + ys[2])*fw + xs[0]];
		result += sobelFilter[25] * volume[(zs[2] * fh + ys[2])*fw + xs[1]];
		result += sobelFilter[26] * volume[(zs[2] * fh + ys[2])*fw + xs[2]];
		float grad = abs(result / 8.0);
		int index = min(int(grad),300);
		unsigned char r = d_clrvec[index * 3], g = d_clrvec[index * 3+1] , b = d_clrvec[index * 3+2];;
		gradient[offset] = grad;
		//val1 = volume[(zs[2] * fh + ys[1])*fw + xs[1]] - volume[(zs[0] * fh + ys[1])*fw + xs[1]];
		float alpha = min(1.0,double(grad / 300));
		//rgb_showFrame[offset3] = rgbframe[offset3] * (1-alpha) + r*alpha;
		//rgb_showFrame[offset3 + 1] = rgbframe[offset3 + 1] * (1 - alpha) + g*alpha;
		//rgb_showFrame[offset3 + 2] = rgbframe[offset3 + 2] * (1 - alpha) + b*alpha;
	}
}

__global__ void FastSobel_RGB( unsigned char* volume, int z0, float* gradient, unsigned char* rgb_showFrame)
{
	int buffLen = d_buffLen[0];
	int fw = d_framewidth[0];
	int fh = d_frameheight[0];

	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	const int z_1 = (z0 - 1 + buffLen) % buffLen;
	const int z1 = (z0 + 1) % buffLen;
	const int xs[3] = { x - 1, x, x + 1 };
	const int ys[3] = { y - 1, y, y + 1 };
	const int zs[3] = { z_1, z0, z1 };
	unsigned char* rgbframe = volume + z0*fh*fw*3;

	if (x>1 && y>1 && x < fw - 1 && y < fh - 1)
	{
		int offset = y*fw + x;
		int offset3 = offset * 3;
		//rgbframe[offset3] =  abs(rframe[offset]);
		//rgbframe[offset3 + 1] = abs(gframe[offset ]);
		//rgbframe[offset3 + 2] = 0;
		int result = 0, result1 = 0,result2=0;
		float val1 = 0;

		result += sobelFilter[0] * volume[((zs[0] * fh + ys[0])*fw + xs[0]) * 3];
		result += sobelFilter[1] * volume[((zs[0] * fh + ys[0])*fw + xs[1]) * 3];
		result += sobelFilter[2] * volume[((zs[0] * fh + ys[0])*fw + xs[2]) * 3];
		result += sobelFilter[3] * volume[((zs[0] * fh + ys[1])*fw + xs[0]) * 3];
		result += sobelFilter[4] * volume[((zs[0] * fh + ys[1])*fw + xs[1]) * 3];
		result += sobelFilter[5] * volume[((zs[0] * fh + ys[1])*fw + xs[2]) * 3];
		result += sobelFilter[6] * volume[((zs[0] * fh + ys[2])*fw + xs[0]) * 3];
		result += sobelFilter[7] * volume[((zs[0] * fh + ys[2])*fw + xs[1]) * 3];
		result += sobelFilter[8] * volume[((zs[0] * fh + ys[2])*fw + xs[2]) * 3];

		result += sobelFilter[18] * volume[((zs[2] * fh + ys[0])*fw + xs[0]) * 3];
		result += sobelFilter[19] * volume[((zs[2] * fh + ys[0])*fw + xs[1]) * 3];
		result += sobelFilter[20] * volume[((zs[2] * fh + ys[0])*fw + xs[2]) * 3];
		result += sobelFilter[21] * volume[((zs[2] * fh + ys[1])*fw + xs[0]) * 3];
		result += sobelFilter[22] * volume[((zs[2] * fh + ys[1])*fw + xs[1]) * 3];
		result += sobelFilter[23] * volume[((zs[2] * fh + ys[1])*fw + xs[2]) * 3];
		result += sobelFilter[24] * volume[((zs[2] * fh + ys[2])*fw + xs[0]) * 3];
		result += sobelFilter[25] * volume[((zs[2] * fh + ys[2])*fw + xs[1]) * 3];
		result += sobelFilter[26] * volume[((zs[2] * fh + ys[2])*fw + xs[2]) * 3];

		result1 += sobelFilter[0] * volume[((zs[0] * fh + ys[0])*fw + xs[0]) * 3 + 1];
		result1 += sobelFilter[1] * volume[((zs[0] * fh + ys[0])*fw + xs[1]) * 3 + 1];
		result1 += sobelFilter[2] * volume[((zs[0] * fh + ys[0])*fw + xs[2]) * 3 + 1];
		result1 += sobelFilter[3] * volume[((zs[0] * fh + ys[1])*fw + xs[0]) * 3 + 1];
		result1 += sobelFilter[4] * volume[((zs[0] * fh + ys[1])*fw + xs[1]) * 3 + 1];
		result1 += sobelFilter[5] * volume[((zs[0] * fh + ys[1])*fw + xs[2]) * 3 + 1];
		result1 += sobelFilter[6] * volume[((zs[0] * fh + ys[2])*fw + xs[0]) * 3 + 1];
		result1 += sobelFilter[7] * volume[((zs[0] * fh + ys[2])*fw + xs[1]) * 3 + 1];
		result1 += sobelFilter[8] * volume[((zs[0] * fh + ys[2])*fw + xs[2]) * 3 + 1];

		result1 += sobelFilter[18] * volume[((zs[2] * fh + ys[0])*fw + xs[0]) * 3 + 1];
		result1 += sobelFilter[19] * volume[((zs[2] * fh + ys[0])*fw + xs[1]) * 3 + 1];
		result1 += sobelFilter[20] * volume[((zs[2] * fh + ys[0])*fw + xs[2]) * 3 + 1];
		result1 += sobelFilter[21] * volume[((zs[2] * fh + ys[1])*fw + xs[0]) * 3 + 1];
		result1 += sobelFilter[22] * volume[((zs[2] * fh + ys[1])*fw + xs[1]) * 3 + 1];
		result1 += sobelFilter[23] * volume[((zs[2] * fh + ys[1])*fw + xs[2]) * 3 + 1];
		result1 += sobelFilter[24] * volume[((zs[2] * fh + ys[2])*fw + xs[0]) * 3 + 1];
		result1 += sobelFilter[25] * volume[((zs[2] * fh + ys[2])*fw + xs[1]) * 3 + 1];
		result1 += sobelFilter[26] * volume[((zs[2] * fh + ys[2])*fw + xs[2]) * 3 + 1];

		result2 += sobelFilter[0] * volume[((zs[0] * fh + ys[0])*fw + xs[0]) * 3 + 2];
		result2 += sobelFilter[1] * volume[((zs[0] * fh + ys[0])*fw + xs[1]) * 3 + 2];
		result2 += sobelFilter[2] * volume[((zs[0] * fh + ys[0])*fw + xs[2]) * 3 + 2];
		result2 += sobelFilter[3] * volume[((zs[0] * fh + ys[1])*fw + xs[0]) * 3 + 2];
		result2 += sobelFilter[4] * volume[((zs[0] * fh + ys[1])*fw + xs[1]) * 3 + 2];
		result2 += sobelFilter[5] * volume[((zs[0] * fh + ys[1])*fw + xs[2]) * 3 + 2];
		result2 += sobelFilter[6] * volume[((zs[0] * fh + ys[2])*fw + xs[0]) * 3 + 2];
		result2 += sobelFilter[7] * volume[((zs[0] * fh + ys[2])*fw + xs[1]) * 3 + 2];
		result2 += sobelFilter[8] * volume[((zs[0] * fh + ys[2])*fw + xs[2]) * 3 + 2];

		result2 += sobelFilter[18] * volume[((zs[2] * fh + ys[0])*fw + xs[0]) * 3 + 2];
		result2 += sobelFilter[19] * volume[((zs[2] * fh + ys[0])*fw + xs[1]) * 3 + 2];
		result2 += sobelFilter[20] * volume[((zs[2] * fh + ys[0])*fw + xs[2]) * 3 + 2];
		result2 += sobelFilter[21] * volume[((zs[2] * fh + ys[1])*fw + xs[0]) * 3 + 2];
		result2 += sobelFilter[22] * volume[((zs[2] * fh + ys[1])*fw + xs[1]) * 3 + 2];
		result2 += sobelFilter[23] * volume[((zs[2] * fh + ys[1])*fw + xs[2]) * 3 + 2];
		result2 += sobelFilter[24] * volume[((zs[2] * fh + ys[2])*fw + xs[0]) * 3 + 2];
		result2 += sobelFilter[25] * volume[((zs[2] * fh + ys[2])*fw + xs[1]) * 3 + 2];
		result2 += sobelFilter[26] * volume[((zs[2] * fh + ys[2])*fw + xs[2]) * 3 + 2];


		float grad0 = abs(result / 8.0), grad1 = abs(result1 / 8.0), grad2 = abs(result2 / 8.0);
		//int index = min(int(grad), 300);
		//unsigned char r = d_clrvec[index * 3], g = d_clrvec[index * 3 + 1], b = d_clrvec[index * 3 + 2];;
		float grad = (grad0 + grad1 + grad2) / 3;
		gradient[offset] = grad;
		//val1 = volume[(zs[2] * fh + ys[1])*fw + xs[1]] - volume[(zs[0] * fh + ys[1])*fw + xs[1]];
		float alpha =  min(1.0, double(grad / 300));
        /*
		if (offset > 0 && offset < fh*fw)
		{
            rgb_showFrame[offset3] = rgb_showFrame[offset3]*0.5+ grad0*0.5;
            rgb_showFrame[offset3 + 1] =  rgb_showFrame[offset3+ 1]*0.5+ grad1*0.5;
            rgb_showFrame[offset3 + 2] =  rgb_showFrame[offset3+ 2]*0.5+ grad2*0.5;
		}
        */
	}
}
void CrowdTracker::calcGradient()
{
	debuggingFile << "calcGradient:" << std::endl;
	cudaMemcpyToSymbol(d_buffLen, &buffLen, sizeof(int));
	cudaMemcpyToSymbol(d_tailidx,&tailidx, sizeof(int));
	/*
	cudaChannelFormatDesc desc = cudaCreateChannelDesc<T>();
	cudaSafeCall(cudaBindTexture2D(0, volumeTexture, volume->gpuAt(frameSize*4), &desc, img.cols, img.rows, img.step));
	*/
	dim3 block(32, 32);
	dim3 grid(divUp(frame_width, 32), divUp(frame_height, 32));
	//FastSobel_t << < grid, block >> >(volumeGray->gpu_ptr(), volumeRGB->gpuAt(mididx*frameSizeRGB) , mididx, gradient->gpu_ptr(), d_rgbframedata);
	FastSobel_RGB << < grid, block >> >(volumeRGB->gpu_ptr(), mididx, gradient->gpu_ptr(), d_rgbframedata);
	gpu::GpuMat gradMat(frame_height, frame_width, CV_32FC1, gradient->gpu_ptr());

}

__global__ void getGradPoints(float* gradient, int2* cornerBuff, unsigned char* mask,unsigned char* d_rgbframedata)
{
	int buffLen = d_buffLen[0];
	int fw = d_framewidth[0];
	int fh = d_frameheight[0];
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < fw&&y < fh)
	{
		
		int offset = y*fw + x;
		if (mask[offset])
		{
			float val = gradient[offset];
			if (val > 100)
			{
				int posidx = atomicAdd(d_total, 1);
				cornerBuff[posidx].x = x;
				cornerBuff[posidx].y = y;
				//d_rgbframedata[offset * 3] = 0;
				//d_rgbframedata[offset * 3 + 1] = 128;
				//d_rgbframedata[offset * 3 + 2] = 255;
			}
		}
	}
}

__global__ void pick_GradPoints(unsigned char* pointRange, int2* cornerBuff,int2* corners)
{
	int addidx = blockIdx.x;
	int ptidx = threadIdx.x;
	int h = d_frameheight[0], w = d_framewidth[0];
	__shared__ unsigned char good[1];
	good[0] = 0;
	__syncthreads();
	if (d_total[0] < blockDim.x)
	{
		int x0 = cornerBuff[addidx].x,y0=cornerBuff[addidx].y;
		if (ptidx < d_total[0])
		{
			int xi = corners[ptidx].x, yi = corners[ptidx].y;
			int dist = abs(x0 - xi) + abs(y0 - yi);
			int range = pointRange[yi*w+xi];
			if (dist < range)
			{  
				good[0]=1;
			}
		}
		__syncthreads();
		if (threadIdx.x==0&&!good[0])
		{
			int posidx = atomicAdd(d_total, 1);
			if (posidx < blockDim.x)
			{
				corners[posidx].x = x0;
				corners[posidx].y = y0;
			}
		}
	}

}

void CrowdTracker::findPoints()
{
	dim3 block(32, 32);
	dim3 grid(divUp(frame_width, 32), divUp(frame_height, 32));
	totalGradPoint = 0;
	
	cudaMemcpyToSymbol(d_total, &totalGradPoint, sizeof(int));
	getGradPoints <<< grid, block >>>(gradient->gpu_ptr(), cornerBuff->gpu_ptr(), mask->gpu_ptr(), d_rgbframedata);
	
	cudaMemcpyFromSymbol(&totalGradPoint, d_total, sizeof(int));
	debuggingFile << "totalGradPoint:"<<totalGradPoint << std::endl;
	cornerBuff->SyncD2H();
	
	acceptPtrNum = 0;
	
	unsigned char* rangeMat = detector->rangeMat.data;
	int2* tmpptr = cornerBuff->cpu_ptr();
	int addpos = 0;
	int2* addptr = corners->cpu_ptr();
	float fp2[2];
	for (int i = 0; i<totalGradPoint; i++)
	{
		int x = tmpptr->x, y = tmpptr->y;
		tmpptr ++;
		uchar range = rangeMat[y*frame_width+x];
		int rangeval = range*range;
		bool good = true;
		int2* ptr2 = corners->cpu_ptr();
		for (int j = 0; j<addpos; j++)
		{
			int x1 = ptr2->x, y1 = ptr2->y;
			int dx = x1 - x, dy = y1 - y;
			if (dx*dx<rangeval&&dy*dx<rangeval)    
			{
				goto findNewPoint;
			}
			ptr2++;
		}
		memcpy(addptr, tmpptr,sizeof(int2));
		addptr++;
		addpos++;
		if (addpos >= nSearch)
			break;
findNewPoint:
	}
	acceptPtrNum = addpos;
	corners->SyncH2D();
	debuggingFile << "acceptPtrNum:" << acceptPtrNum << std::endl;
	/*
	if (totalGradPoint>0 && totalGradPoint<65535)
	{
		cudaMemcpyToSymbol(d_total, &acceptPtrNum, sizeof(int));
		debuggingFile << "totalGradPoint : " << totalGradPoint << std::endl;
		pick_GradPoints << < totalGradPoint, nSearch >> >(pointRange->gpu_ptr(), cornerBuff->gpu_ptr(), corners->gpu_ptr());
		cudaMemcpyFromSymbol(&acceptPtrNum, d_total, sizeof(int));
		corners->SyncD2H();
	}

	debuggingFile << "total Find: " << acceptPtrNum << std::endl;
	*/
	
}

__global__ void filterTracks(TracksInfo trkinfo, uchar* status, float2* update_ptr, float* d_persMap)
{
	int idx = threadIdx.x;
	int len = trkinfo.lenVec[idx];
	bool flag = status[idx];
	float x = update_ptr[idx].x, y = update_ptr[idx].y;
	int frame_width = d_framewidth[0], frame_heigh = d_frameheight[0];
	trkinfo.nextTrkptr[idx].x = x;
	trkinfo.nextTrkptr[idx].y = y;
	float curx = trkinfo.curTrkptr[idx].x, cury = trkinfo.curTrkptr[idx].y;
	float dx = x - curx, dy = y - cury;
	float dist = sqrt(dx*dx + dy*dy);
	float cumDist = dist + trkinfo.curDistPtr[idx];
	trkinfo.nextDistPtr[idx] = cumDist;
	if (flag&&len>0)
	{

		int xb = x + 0.5, yb = y + 0.5;
		UperLowerBound(xb, 0, frame_width-1);
		UperLowerBound(yb, 0, frame_heigh - 1);
		float persval = d_persMap[yb*frame_width + xb];
		if (xb < 10 && yb < 10)flag = false;
        float avglen=cumDist/len;
        //if((dist-avglen)/(dist+avglen)>0.01)flag=false;
		//        int prex=trkinfo.curTrkptr[idx].x+0.5, prey=trkinfo.curTrkptr[idx].y+0.5;
		//        int trkdist=abs(prex-xb)+abs(prey-yb);
		float trkdist = abs(dx) + abs(dy);
        //if (trkdist>persval/5)
        if (trkdist>100)
		{
			flag = false;
		}
		//printf("%d,%.2f,%d|",trkdist,persval,flag);
		int Movelen = 150 / sqrt(persval);
		Movelen = 15;
		//Movelen is the main factor wrt perspective
		//        printf("%d\n",Movelen);

		if (flag&&Movelen<len)
		{
			//            int offset = (tailidx+bufflen-Movelen)%bufflen;
			//            FeatPts* dataptr = next_ptr-tailidx*NQue;
			//            FeatPts* aptr = dataptr+offset*NQue;
			//            float xa=aptr[idx].x,ya=aptr[idx].y;
			FeatPts* ptr = trkinfo.getPtr_(trkinfo.trkDataPtr, idx, Movelen);
			float xa = ptr->x, ya = ptr->y;
			float displc = sqrt((x - xa)*(x - xa) + (y - ya)*(y - ya));
			float curveDist = cumDist - *(trkinfo.getPtr_(trkinfo.distDataPtr, idx, Movelen));

            trkinfo.curveDataPtr[idx]=displc/(curveDist+0.1)*255;
			//if(persval*0.1>displc)
			if ( displc<3)
			{
				flag = false;
			}
		}
	}
	int newlen = flag*(len + (len<trkinfo.buffLen));
	trkinfo.lenVec[idx] = newlen;
	if (newlen>minTrkLen)
	{
		FeatPts* pre_ptr = trkinfo.preTrkptr;
		float prex = pre_ptr[idx].x, prey = pre_ptr[idx].y;
		float vx = (x - prex) / minTrkLen, vy = (y - prey) / minTrkLen;
		float spd = sqrt(vx*vx + vy*vy);
		trkinfo.nextSpdPtr[idx] = spd;
		trkinfo.nextVeloPtr[idx].x = vx, trkinfo.nextVeloPtr[idx].y = vy;
        trkinfo.cumVeloPtr[idx].y=(vy+trkinfo.cumVeloPtr[idx].y)/2;
        trkinfo.cumVeloPtr[idx].x=(vx+trkinfo.cumVeloPtr[idx].x)/2;
	}
    else if(newlen==0)
    {
        trkinfo.nextDistPtr[idx]=0;
        trkinfo.curveDataPtr[idx]=0;
    }
}
__global__ void removeDupTracks(TracksInfo trkInfo,float* persmap)
{
    int c = threadIdx.x, r = blockIdx.x;
    int clen = trkInfo.lenVec[c], rlen = trkInfo.lenVec[r];
    if (r<c&&clen&&rlen)
    {
        FeatPts* cur_ptr = trkInfo.curTrkptr;
        float cx1 = cur_ptr[c].x, cy1 = cur_ptr[c].y;
        float rx1 = cur_ptr[r].x, ry1 = cur_ptr[r].y;
        int dx = abs(rx1 - cx1), dy = abs(ry1 - cy1);
        if(dx<1&&dy<1)
        {
            if(clen>rlen)
            {
                trkInfo.lenVec[rlen]=0;
            }
            else
            {
                trkInfo.lenVec[clen]=0;
            }
        }
    }
}
void CrowdTracker::filterTrackGPU()
{
	debuggingFile << "filterTrackGPU" << std::endl;
	trkInfo = tracksGPU->getInfoGPU();
	trkInfo.preTrkptr = trkInfo.getVec_(trkInfo.trkDataPtr, minTrkLen - 1);
	
	filterTracks <<< 1, nFeatures >>>(trkInfo, gpuStatus.data, (float2 *)gpuNextPts.data, persMap->gpu_ptr());
    removeDupTracks<<<nFeatures,nFeatures>>>(trkInfo, persMap->gpu_ptr());
	tracksGPU->increPtr();
	trkInfo = tracksGPU->getInfoGPU();
	trkInfo.preTrkptr = trkInfo.getVec_(trkInfo.trkDataPtr, minTrkLen);
	debuggingFile << "Finshed filterTrackGPU" << std::endl;
}

__global__ void  addNewPts(FeatPts* cur_ptr, int* lenVec, int2* new_ptr, float2* nextPtrs)
{
	int idx = threadIdx.x;
	int dim = blockDim.x;
	__shared__ int counter[1];
	counter[0] = 0;
	__syncthreads();
	//printf("(%d)", idx);
	if (lenVec[idx] <= 0)
	{
		int posidx = atomicAdd(counter, 1);
		
		if (posidx<dim)
		{
			float x = new_ptr[posidx].x, y = new_ptr[posidx].y;
			cur_ptr[idx].x = x;
			cur_ptr[idx].y = y;
			lenVec[idx] += 1;
			
		}
	}

	nextPtrs[idx].x = cur_ptr[idx].x;
	nextPtrs[idx].y = cur_ptr[idx].y;
	//__syncthreads();
	//d_total[0] = counter[0];
}

__global__ void applyPointPersMask(unsigned char* d_mask, FeatPts* cur_ptr, int* lenVec, float* d_persMap)
{
	int pidx = blockIdx.x;
	int len = lenVec[pidx];
	if (len>0)
	{
		float px = cur_ptr[pidx].x, py = cur_ptr[pidx].y;
		int blocksize = blockDim.x;
		int w = d_framewidth[0], h = d_frameheight[0];
		int localx = threadIdx.x, localy = threadIdx.y;
		int pxint = px + 0.5, pyint = py + 0.5;
		UperLowerBound(pyint, 0, h - 1);
		UperLowerBound(pxint, 0, w - 1);
		float persval = d_persMap[pyint*w + pxint];
		float range = Pers2Range(persval)+1;
		int offset = range + 0.5;
		int yoffset = localy - blocksize / 2;
		int xoffset = localx - blocksize / 2;
		
		if (abs(yoffset)<range&&abs(xoffset)<range)
		{
			int globalx = xoffset + pxint, globaly = yoffset + pyint;
			int globaloffset = globaly*w + globalx;
			if (globaloffset < w*h && globaloffset>0)
			{
				//printf("%d)", globaloffset);
				d_mask[globaloffset] = 0;
			}
		}
	}
}
void CrowdTracker::PersExcludeMask()
{
	addNewPts << <1, nFeatures, 0, cornerStream >> >(tracksGPU->curTrkptr, tracksGPU->lenVec, corners->gpu_ptr(), (float2*)gpuPrePts.data);
	cudaMemcpyAsync(mask->gpu_ptr(), roimask->gpu_ptr(), frame_height*frame_width*sizeof(unsigned char), cudaMemcpyDeviceToDevice, cornerStream);
	dim3 block(32, 32, 1);
	applyPointPersMask << <nFeatures, block, 0, cornerStream >> >(mask->gpu_ptr(), tracksGPU->curTrkptr, tracksGPU->lenVec, persMap->gpu_ptr());
	//corners->SyncD2HStream(cornerStream);
	/*
	addNewPts << <1, nFeatures >> >(tracksGPU->curTrkptr, tracksGPU->lenVec, corners->gpu_ptr(), (float2*)gpuPrePts.data);
	debuggingFile << "there" << std::endl;
	std::cout << std::endl;
	cudaMemcpy(mask->gpu_ptr(), roimask->gpu_ptr(), frame_height*frame_width*sizeof(unsigned char), cudaMemcpyDeviceToDevice);
	dim3 block(32, 32, 1);
	applyPointPersMask << <nFeatures, block>> >(mask->gpu_ptr(), tracksGPU->curTrkptr, tracksGPU->lenVec, persMap->gpu_ptr());
	*/
}
__global__ void renderFrame(unsigned char* d_renderMask,unsigned char* d_frameptr,int totallen,unsigned char* d_mask)
{
    int offset=(blockIdx.x*blockDim.x+threadIdx.x)*3;
    int offsetp=blockIdx.x*blockDim.x+threadIdx.x;
    int maskval = d_renderMask[offset];
    if(offsetp<totallen)
    {
        /*
        if (d_renderMask[offset] || d_renderMask[offset + 1] || d_renderMask[offset + 2])
        {
            d_frameptr[offset] = d_frameptr[offset] * 0.5 + d_renderMask[offset] * 0.5;
            d_frameptr[offset + 1] = d_frameptr[offset + 1] * 0.5 + d_renderMask[offset + 1] * 0.5;
            d_frameptr[offset + 2] = d_frameptr[offset + 2] * 0.5 + d_renderMask[offset + 2] * 0.5;
        }
        else
        {
            d_frameptr[offset] = d_frameptr[offset] * 0.5 ;
            d_frameptr[offset + 1] = d_frameptr[offset + 1] * 0.5 ;
            d_frameptr[offset + 2] = d_frameptr[offset + 2] * 0.5 ;
        }
        */
        d_frameptr[offset] = d_frameptr[offset] * 0.5 + d_renderMask[offset] * 0.5;
        d_frameptr[offset + 1] = d_frameptr[offset + 1] * 0.5 + d_renderMask[offset + 1] * 0.5;
        d_frameptr[offset + 2] = d_frameptr[offset + 2] * 0.5 + d_renderMask[offset + 2] * 0.5;
//        d_frameptr[offset]=d_renderMask[offset]*0.5;
//        d_frameptr[offset+1]=d_renderMask[offset+1]*0.5;
//        d_frameptr[offset+2]=d_renderMask[offset+2]*0.5;
        /*
        if (d_mask[offsetp])
        {
            d_frameptr[offset] = d_frameptr[offset]*0.5;
            d_frameptr[offset + 1] = d_frameptr[offset + 1] * 0.5;
            d_frameptr[offset + 2] = d_frameptr[offset + 2] * 0.5;
        }
        */
    }
}
__global__ void renderGroup(unsigned char* d_renderMask,FeatPts* cur_ptr,int* lenVec,unsigned char* d_clrvec,float* d_persMap,int* d_neighbor)
{
    int pidx=blockIdx.x;
    int len=lenVec[pidx];
    if(len>0)
    {
        float px=cur_ptr[pidx].x,py=cur_ptr[pidx].y;
        int blocksize = blockDim.x;
        int w=d_framewidth[0],h=d_frameheight[0];
        int localx = threadIdx.x,localy=threadIdx.y;
        int pxint = px+0.5,pyint = py+0.5;
        float persval =d_persMap[pyint*w+pxint];
        float range=Pers2Range(persval);
        int offset=range+0.5;
        int yoffset = localy-blocksize/2;
        int xoffset = localx-blocksize/2;
        if(abs(yoffset)<range&&abs(xoffset)<range)
        {
            int globalx=xoffset+pxint,globaly=yoffset+pyint;
            int globalOffset=(globaly*w+globalx)*3;
            d_renderMask[globalOffset]=255;
            d_renderMask[globalOffset+1]=0;
            d_renderMask[globalOffset+2]=0;
        }
    }
}

__global__ void renderGray(unsigned char* framePtr,int* grayPtr,uchar* clrvec,int totallen,float minval,float maxval)
{
    int offset=(blockIdx.x*blockDim.x+threadIdx.x)*3;
    int offsetp=blockIdx.x*blockDim.x+threadIdx.x;
    if(offsetp<totallen)
    {
        int val = grayPtr[offsetp];
        if(val)
        {
            framePtr[offset]=clrvec[val*3];
            framePtr[offset+1]=clrvec[val*3+1];
            framePtr[offset+2]=clrvec[val*3+2];
        }
        //framePtr[offset+1]=0;
        //framePtr[offset+2]=0;
        //framePtr[offset+1]=framePtr[offset+1]*(val);
        //framePtr[offset+2]=framePtr[offset+2]*(val);
    }
}
void CrowdTracker::Render(unsigned char* framedata)
{

   dim3 blockSize(32,32,1);
   tracksGPU->lenData->SyncD2H();
   int val=0;
   for(int i=0;i<nFeatures;i++)
   {
       val+=(tracksGPU->lenData->cpu_ptr()[i]>0);
   }
   debuggingFile<<"Render:"<<val<<std::endl;
   //renderGroup<<<nFeatures,blockSize>>>(renderMask->gpu_ptr(),tracksGPU->curTrkptr,tracksGPU->lenVec,clrvec->gpu_ptr(),persMap->gpu_ptr(),nbCount->gpu_ptr());

   int nblocks = (frame_height*frame_width)/1024;
   //renderMask->toZeroD();
   renderGray<<<nblocks,nFeatures>>>(renderMask->gpu_ptr(),labelMap->gpu_ptr(),clrvec->gpu_ptr(),frameSizeRGB,headMin,headMax);
   renderFrame<<<nblocks,nFeatures>>>(renderMask->gpu_ptr(),rgbMat.data,frame_width*frame_height,mask->gpu_ptr());

   renderMask->toZeroD();
   //cudaMemcpy(rgbMat.data,framedata,frameSizeRGB,cudaMemcpyHostToDevice);

   cudaMemcpy(framedata,rgbMat.data,frameSizeRGB,cudaMemcpyDeviceToHost);
}

__global__ void rgb2grayKernel(unsigned char * d_frameRGB, unsigned char* d_frameGray,int total)
{
	int offset = blockIdx.x*blockDim.x + threadIdx.x;
	int offset3 = offset*3;
	if (offset < total)
	{
		float r = d_frameRGB[offset3], g = d_frameRGB[offset3+1],b=d_frameRGB[offset3+2];
		d_frameGray[offset] = 0.299*r + 0.587*g + 0.114*b;
	}
}
void CrowdTracker::RGB2Gray(unsigned char * d_frameRGB,unsigned char* d_frameGray)
{
	int nblocks = frameSize / 1024+1;
	rgb2grayKernel << <nblocks, 1024 >> >(d_frameRGB, d_frameGray, frameSize);
}
__global__ void clearLostStats(int* lenVec, int* d_neighbor, float* d_cosine, float* d_velo, float* d_distmat, int nFeatures)
{
	int c = threadIdx.x, r = blockIdx.x;
	if (r<nFeatures, c<nFeatures)
	{
		bool flag1 = (lenVec[c]>0), flag2 = (lenVec[r]>0);
		bool flag = flag1&&flag2;
		if (!flag)
		{

			d_neighbor[r*nFeatures + c] = 0;
			d_neighbor[c*nFeatures + r] = 0;
			d_cosine[r*nFeatures + c] = 0;
			d_cosine[c*nFeatures + r] = 0;
			d_velo[r*nFeatures + c] = 0;
			d_velo[c*nFeatures + r] = 0;
			d_distmat[r*nFeatures + c] = 0;
			d_distmat[c*nFeatures + r] = 0;
		}
	}
}
__global__ void searchNeighbor(TracksInfo trkinfo,
	int* d_neighbor, float* d_cosine, float* d_velo, float* d_distmat,
	float * d_persMap, int nFeatures)
{
	int c = threadIdx.x, r = blockIdx.x;
	int clen = trkinfo.lenVec[c], rlen = trkinfo.lenVec[r];
    FeatPts* cur_ptr = trkinfo.curTrkptr;
	if (clen>minTrkLen&&rlen>minTrkLen&&r<c)
	{
		//        int offset = (tailidx+bufflen-minTrkLen)%bufflen;
		//        FeatPts* pre_ptr=data_ptr+NQue*offset;
		//        FeatPts* pre_ptr=trkinfo.preTrkptr;//trkinfo.getVec_(trkinfo.trkDataPtr,minTrkLen-1);
		//        float cx0=pre_ptr[c].x,cy0=pre_ptr[c].y;
        //        float rx0=pre_ptr[r].x,ry0=pre_ptr[r].y;
		float cx1 = cur_ptr[c].x, cy1 = cur_ptr[c].y;
		float rx1 = cur_ptr[r].x, ry1 = cur_ptr[r].y;
		float dx = abs(rx1 - cx1), dy = abs(ry1 - cy1);
		float dist = sqrt(dx*dx + dy*dy);
		int  ymid = (ry1 + cy1) / 2.0 + 0.5, xmid = (rx1 + cx1) / 2.0 + 0.5;
		float persval = 0;
		int ymin = min(ry1, cy1), xmin = min(rx1, cx1);
		persval = d_persMap[ymin*d_framewidth[0] + xmin];
		float hrange = persval, wrange = persval;
		if (hrange<2)hrange = 2;
		if (wrange<2)wrange = 2;
        //if(dx<wrange&&dy<hrange)
        {
            float distdecay = 0.05, cosdecay = 0.1, velodecay = 0.05;
            /*
            float vx0 = rx1 - rx0, vx1 = cx1 - cx0, vy0 = ry1 - ry0, vy1 = cy1 - cy0;
            float norm0 = sqrt(vx0*vx0 + vy0*vy0), norm1 = sqrt(vx1*vx1 + vy1*vy1);
            float veloCo = abs(norm0-norm1)/(norm0+norm1);
            float cosine = (vx0*vx1 + vy0*vy1) / norm0 / norm1;
            */
            float vrx = trkinfo.curVeloPtr[r].x, vry = trkinfo.curVeloPtr[r].y
                , vcx = trkinfo.curVeloPtr[c].x, vcy = trkinfo.curVeloPtr[c].y;
            float normr = trkinfo.curSpdPtr[r], normc = trkinfo.curSpdPtr[c];
            float veloCo = abs(normr - normc) / (normr + normc);
            float cosine = (vrx*vcx + vry*vcy) / normr / normc;
            dist = wrange*1.5 / (dist + 0.01);
            dist = 2 * dist / (1 + abs(dist)) - 1;
            //dist=-((dist > wrange) - (dist < wrange));
            d_distmat[r*nFeatures + c] = dist + d_distmat[r*nFeatures + c] * (1 - distdecay);
            d_distmat[c*nFeatures + r] = dist + d_distmat[c*nFeatures + r] * (1 - distdecay);
            d_cosine[r*nFeatures + c] = cosine + d_cosine[r*nFeatures + c] * (1 - cosdecay);
            d_cosine[c*nFeatures + r] = cosine + d_cosine[c*nFeatures + r] * (1 - cosdecay);
            d_velo[r*nFeatures + c] = veloCo + d_velo[r*nFeatures + c] * (1 - velodecay);
            d_velo[c*nFeatures + r] = veloCo + d_velo[c*nFeatures + r] * (1 - velodecay);
            if (d_distmat[r*nFeatures + c]>5 && d_cosine[r*nFeatures + c]>2&&cosine>0.5)//&&d_velo[r*nFeatures+c]<(14*velodecay)*0.9)
            {
                d_neighbor[r*nFeatures + c] += 1;
                d_neighbor[c*nFeatures + r] += 1;
            }
            else
            {
                d_neighbor[r*nFeatures + c] =0;
                d_neighbor[c*nFeatures + r] =0;
            }
        }
	}
}
__global__ void searchNeighborLoop(TracksInfo trkinfo,
    int* d_neighbor, float* d_cosine, float* d_velo, float* d_distmat,
    float * d_persMap, int nFeatures)
{
    int c = threadIdx.x, r = blockIdx.x;
    int clen = trkinfo.lenVec[c], rlen = trkinfo.lenVec[r];

    if (clen>minTrkLen&&rlen>minTrkLen&&r<c)
    {
        bool isNeighbor=true;
        for(int i=0;i<minTrkLen;i++)
        {
            FeatPts* cur_ptr = trkinfo.getVec_(trkinfo.trkDataPtr,i);
            float cx1 = cur_ptr[c].x, cy1 = cur_ptr[c].y;
            float rx1 = cur_ptr[r].x, ry1 = cur_ptr[r].y;
            float dx = abs(rx1 - cx1), dy = abs(ry1 - cy1);
            float dist = sqrt(dx*dx + dy*dy);
            int  ymid = (ry1 + cy1) / 2.0 + 0.5, xmid = (rx1 + cx1) / 2.0 + 0.5;
            float persval = 0;
            int ymin = min(ry1, cy1), xmin = min(rx1, cx1);
            persval = d_persMap[ymin*d_framewidth[0] + xmin];
            float hrange = persval, wrange = persval;
            if (hrange<2)hrange = 2;
            if (wrange<2)wrange = 2;
            float vrx = trkinfo.curVeloPtr[r].x, vry = trkinfo.curVeloPtr[r].y
                , vcx = trkinfo.curVeloPtr[c].x, vcy = trkinfo.curVeloPtr[c].y;
            float normr = trkinfo.curSpdPtr[r], normc = trkinfo.curSpdPtr[c];
            float veloCo = abs(normr - normc) / (normr + normc);
            float cosine = (vrx*vcx + vry*vcy) / normr / normc;
//            float vx = vrx-vcx;
//            float vy = vry-vcy;
//            float vdiff = sqrt(vx*vx+vy*vy);
            if (!(dx<wrange&&dy<hrange&&cosine>0.6&&veloCo<0.2))
            {
                isNeighbor=false;
                break;
            }
        }
        d_neighbor[r*nFeatures + c] = isNeighbor;
        d_neighbor[c*nFeatures + r] = isNeighbor;
    }
}

void CrowdTracker::pointCorelate()
{
	clearLostStats << <nFeatures, nFeatures >> >(tracksGPU->lenData->gpu_ptr(),nbCount->gpu_ptr(), cosCo->gpu_ptr(), veloCo->gpu_ptr(), distCo->gpu_ptr(), nFeatures);
    searchNeighborLoop << <nFeatures, nFeatures >> >(trkInfo, nbCount->gpu_ptr(), cosCo->gpu_ptr(), veloCo->gpu_ptr(), distCo->gpu_ptr(), persMap->gpu_ptr(), nFeatures);
}

__global__ void  makeGroupKernel(int* labelidx,Groups groups,TracksInfo trkinfo)
{
    int pidx=threadIdx.x;
    int gidx=blockIdx.x;
    int* idx_ptr=groups.trkPtsIdxPtr;
    int* count_ptr=groups.ptsNumPtr;
    int nFeatures=groups.trkPtsNum;
    int* cur_gptr = idx_ptr+gidx*nFeatures;
    FeatPts* cur_Trkptr=trkinfo.curTrkptr+pidx;
    float2* cur_veloPtr=trkinfo.curVeloPtr+pidx;
    __shared__ int counter;
    __shared__ float com[2],velo[2];
    __shared__ int left,right,top,bot;
    left=9999,right=0,top=9999,bot=0;
    com[0]=0,com[1]=0;
    velo[0]=0,velo[1]=0;
    counter=0;
    __syncthreads();
    if(labelidx[pidx]==gidx)
    {
        float x=cur_Trkptr->x,y=cur_Trkptr->y;
        int px=x+0.5,py=y+0.5;
        int pos=atomicAdd(&counter,1);
        cur_gptr[pos]=pidx;
        atomicAdd(com,x);
        atomicAdd((com+1),y);
        atomicAdd(velo,cur_veloPtr->x);
        atomicAdd((velo+1),cur_veloPtr->y);
        atomicMin(&left,px);
        atomicMin(&top,py);
        atomicMax(&right,px);
        atomicMax(&bot,py);
    }
    __syncthreads();
    if(threadIdx.x==0)
    {
        count_ptr[gidx]=counter;
        groups.comPtr[gidx].x=com[0]/counter;
        groups.comPtr[gidx].y=com[1]/counter;
        groups.veloPtr[gidx].x=velo[0]/counter;
        groups.veloPtr[gidx].y=velo[1]/counter;
        groups.bBoxPtr[gidx].left=left;
        groups.bBoxPtr[gidx].top=top;
        groups.bBoxPtr[gidx].right=right;
        groups.bBoxPtr[gidx].bottom=bot;
        float area=(bot-top)*(right-left);
        groups.areaPtr[gidx]=area;
        //printf("%d,%f/",gidx,area);
    }
}
__global__ void  groupProp(int* labelidx,Groups groups,TracksInfo trkinfo)
{

}
__host__ __device__ __forceinline__ float cross_(const cvxPnt& O, const cvxPnt& A, const cvxPnt &B)
{
    return (A.x - O.x) * (B.y - O.y) - (A.y - O.y) * (B.x - O.x);
}



__global__ void groupUpdatePts(BBox bb,TracksInfo trkinfo,int* counter,int* idxvec)
{
    const int pidx = threadIdx.x;
    FeatPts p=trkinfo.curTrkptr[pidx];
    counter[0]=0;
    if(p.x>=bb.left&&p.x<=bb.right&&p.y>=bb.top&&p.y<=bb.bottom)
    {
        int pos = atomicAdd(counter,1);
        idxvec[pos]=pidx;
    }
}
void CrowdTracker::KLT_updates_Group(int idx)
{

    GroupTrack& gTrk = (*groupsTrk)[idx];
    float2 updates = (*kltUpdateVec)[idx];
    bool useCorr=false;
    if(gTrkCorr[idx].size()>0)
    {

        float2 corrUpdates={0,0};
        int counter=0;
        for(int i=0;i<gTrkCorr[idx].size();i++)
        {
            int corrIdx = gTrkCorr[idx][i];
            if((*updateType)[corrIdx]==0)
            {
                useCorr=true;
                corrUpdates.x+=(*kltUpdateVec)[corrIdx].x;
                corrUpdates.y+=(*kltUpdateVec)[corrIdx].y;
                counter++;
            }
        }
        if(useCorr)
        {
            updates.x = corrUpdates.x/counter;
            updates.y = corrUpdates.y/counter;
        }
    }
    int headstats[CC_STAT_TOTAL];
    int* oldHeadstats = gTrk.getCur_(gTrk.headStats->cpu_ptr(),CC_STAT_TOTAL);
    /*
    int chekSpan=2;

    if(gTrk.len>gTrk.updateCount+chekSpan)
    {
        int* preHeadStats = gTrk.getPtr_(gTrk.headStats->cpu_ptr(),gTrk.updateCount+chekSpan);
        int* curHeadStats = gTrk.getPtr_(gTrk.headStats->cpu_ptr(),gTrk.updateCount);
        if(curHeadStats[CC_STAT_AREA]>0&&preHeadStats[CC_STAT_AREA]>0)
        {
            int xdiff = curHeadStats[CC_STAT_HEAD_X]-preHeadStats[CC_STAT_HEAD_X];
            int ydiff = curHeadStats[CC_STAT_HEAD_Y]-preHeadStats[CC_STAT_HEAD_Y];
            updates.x=float(xdiff)/float(chekSpan);
            updates.y=float(ydiff)/float(chekSpan);
            std::cout<<idx<<"_updates:"<<updates.x<<","<<updates.y<<std::endl;
        }
    }
    */

    BBox oldbbox = *gTrk.getCurBBox();
    BBox newbbox;
    newbbox.left = (*kltUpdateBoxVec)[idx].left + 0.5,
    newbbox.right = (*kltUpdateBoxVec)[idx].right + 0.5,
    newbbox.top = (*kltUpdateBoxVec)[idx].top + 0.5,
    newbbox.bottom = (*kltUpdateBoxVec)[idx].bottom + 0.5;

    memcpy(gTrk.getNext_(gTrk.bBox->cpu_ptr()), &newbbox, sizeof(BBox));
    cudaMemcpy(gTrk.getNext_(gTrk.bBoxPtr), &newbbox, sizeof(BBox), cudaMemcpyHostToDevice);

    float2 newcom;
    newcom.x = gTrk.getCurCom()->x + updates.x;
    newcom.y = gTrk.getCurCom()->y + updates.y;
    memcpy(gTrk.getNext_(gTrk.com->cpu_ptr()), &newcom, sizeof(float2));
    cudaMemcpy(gTrk.getNext_(gTrk.comPtr), &newcom, sizeof(float2), cudaMemcpyHostToDevice);

    memcpy(gTrk.getNext_(gTrk.velo->cpu_ptr()), &updates, sizeof(float2));
    cudaMemcpy(gTrk.getNext_(gTrk.veloPtr), &updates, sizeof(float2), cudaMemcpyHostToDevice);

    memcpy(gTrk.getNext_(gTrk.area->cpu_ptr()), gTrk.getCur_(gTrk.area->cpu_ptr()), sizeof(float));
    cudaMemcpy(gTrk.getNext_(gTrk.areaPtr), gTrk.getCur_(gTrk.areaPtr), sizeof(float), cudaMemcpyDeviceToDevice);

    //roupUpdatePts<<<1,nFeatures>>>(newbbox,trkInfo,gTrk.getNext_(gTrk.ptsNumPtr),gTrk.getNext_(gTrk.trkPtsIdxPtr));
    //gTrk.ptsNum->SyncD2H();
    //gTrk.trkPtsIdx->SyncD2H();

    /*
    memcpy(gTrk.getNext_(gTrk.trkPtsIdx->cpu_ptr()), gTrk.getCur_(gTrk.trkPtsIdx->cpu_ptr()), sizeof(int)*nFeatures);
    cudaMemcpy(gTrk.getNext_(gTrk.trkPtsIdxPtr), gTrk.getCur_(gTrk.trkPtsIdxPtr), sizeof(int)*nFeatures, cudaMemcpyDeviceToDevice);

    memcpy(gTrk.getNext_(gTrk.ptsNum->cpu_ptr()), gTrk.getCur_(gTrk.ptsNum->cpu_ptr()), sizeof(int));
    cudaMemcpy(gTrk.getNext_(gTrk.ptsNumPtr), gTrk.getCur_(gTrk.ptsNumPtr), sizeof(int), cudaMemcpyDeviceToDevice);

    memcpy(gTrk.getNext_(gTrk.trkPts->cpu_ptr()), gTrk.getCur_(gTrk.trkPts->cpu_ptr()), sizeof(float2)*nFeatures);
    cudaMemcpy(gTrk.getNext_(gTrk.trkPtsPtr), gTrk.getCur_(gTrk.trkPtsPtr), sizeof(float2)*nFeatures, cudaMemcpyDeviceToDevice);
    */


            //std::cout<<"klt"<<idx<<","<<oldHeadstats[CC_STAT_LEFT]<<std::endl;
    memcpy(headstats,oldHeadstats,sizeof(int)*CC_STAT_TOTAL);
    if(gTrk.trkType==HEAD_TRK&&oldHeadstats[CC_STAT_AREA]>0)
    {
        headstats[CC_STAT_LEFT]=oldHeadstats[CC_STAT_LEFT]-oldbbox.left+newbbox.left;
        headstats[CC_STAT_TOP]=oldHeadstats[CC_STAT_TOP]-oldbbox.top+newbbox.top;
        headstats[CC_STAT_HEAD_X]=oldHeadstats[CC_STAT_HEAD_X]-oldbbox.left+newbbox.left;
        headstats[CC_STAT_HEAD_Y]=oldHeadstats[CC_STAT_HEAD_Y]-oldbbox.top+newbbox.top;
        memcpy(gTrk.getNext_(gTrk.headStats->cpu_ptr(),CC_STAT_TOTAL),headstats,sizeof(int)*CC_STAT_TOTAL);
        cudaMemcpy(gTrk.getNext_(gTrk.headStats->gpu_ptr(),CC_STAT_TOTAL),headstats,sizeof(int)*CC_STAT_TOTAL,cudaMemcpyHostToDevice);
    }
    float persval=(*persMap)[oldbbox.top*frame_width+oldbbox.left];
    float mindist = persval/100-0.1;
    //std::cout<<idx<<"updates:"<<updates.x<<","<<updates.y<<std::endl;
    float dist=sqrt(updates.x*updates.x+updates.y*updates.y);
    float increment=2;
    if(!useCorr)
        increment=1;
    else
        increment=0.5;
    if(gTrk.trkType==HEAD_TRK&&abs(updates.x)<mindist&&abs(updates.y)<mindist)
        increment=0.1;
    gTrk.updateCount+=increment;
    //gTrk.headStats->SyncH2D();

    gTrk.increPtr();
    (*kltUpdateBoxVec)[idx].left=(*kltUpdateBoxVec)[idx].left+updates.x;
     (*kltUpdateBoxVec)[idx].top=(*kltUpdateBoxVec)[idx].top+updates.y;
     (*kltUpdateBoxVec)[idx].right=(*kltUpdateBoxVec)[idx].right+updates.x;
     (*kltUpdateBoxVec)[idx].bottom=(*kltUpdateBoxVec)[idx].bottom+updates.y;
}
void CrowdTracker::matchGroups()
{
    /** compute Score**/
    std::clock_t start=std::clock();
    /*
    if (groupsTrk->numGroup > 0)
    {
        KLT_update_Group_Kernel <<< groupsTrk->numGroup, nFeatures >>>(groupsTrk->groupTracks->gpu_ptr(), groupsTrk->vacancy->gpu_ptr(),
            trkInfo, kltUpdateVec->gpu_ptr(),kltUpdateBoxVec->gpu_ptr());
        kltUpdateVec->SyncD2H();
        kltUpdateBoxVec->SyncD2H();
    }
    kltUpdateBoxVec->toZeroH();
    */
    //rankCountNew->toZeroH();
    rankCountOld->toZeroH();
    //rankingNew->toZeroH();
    rankingOld->toZeroH();
    //scoreNew->toZeroH();
    scoreOld->toZeroH();
    for(int i=0;i<groupsTrk->numGroup;i++)
    {
        BBox trkBox = *groupsTrk->getCurBBox(i);
        GroupTrack& gTrk = *groupsTrk->getPtr(i);
        float2 com = *gTrk.getCurCom();
        trkBox.left = (*kltUpdateBoxVec)[i].left+0.5;
        trkBox.top = (*kltUpdateBoxVec)[i].top+0.5;
        trkBox.right = (*kltUpdateBoxVec)[i].right+0.5;
        trkBox.bottom = (*kltUpdateBoxVec)[i].bottom+0.5;
        float trkArea=groupsTrk->getCurArea(i);
        int* idxvec=rankingOld->cpuAt(i*nFeatures);
        float* valvec=scoreOld->cpuAt(i*nFeatures);
        int& counter=(*rankCountOld)[i];
        if((*groupsTrk->vacancy)[i])
        {
        for(int j=1;j<groups->numGroups;j++)
        {
            int dettype =(j>=groups->kltGroupNum);
            if(gTrk.trkType==dettype)
            {
                BBox newBox = (*groups->bBox)[j];
                float persVal=(*persMap)[trkBox.top*frame_width+trkBox.left];
                //int bboxw=persVal/2,bboxh=persVal*2;
                float newArea = (*groups->area)[j];

                int minw = min((trkBox.right-newBox.left),(newBox.right-trkBox.left));
                int minh = min((trkBox.bottom-newBox.top),(newBox.bottom-trkBox.top));
                UperLowerBound(minw,0,1920);
                UperLowerBound(minh,0,1080);
                int area = minw*minh;
                float trkCoArea=float(area)/float(trkArea);
                float newCoArea=float(area)/float(newArea);
                float coArea=float(area)/float(newArea+trkArea)*2;
                //float shapeCoef = abs(bboxw-newBox.right+newBox.left)*abs(bboxh-newBox.bottom+newBox.top)/float(bboxw)/float(bboxh);

                float2 detCom = (*groups->com)[j];
                float2 comUpdate = {detCom.x-com.x,detCom.y-com.y};
                float2 kltUpdate = (*kltUpdateVec)[i];
                float normCom = sqrt(comUpdate.x*comUpdate.x+comUpdate.y*comUpdate.y);
                float normKLT = sqrt(kltUpdate.x*kltUpdate.x+kltUpdate.y*kltUpdate.y);
                //float cosine = (comUpdate.x*kltUpdate.x+comUpdate.y*kltUpdate.y);
                bool isGoodUpdate = !(normCom>normKLT*15);//((cosine<=0&&normCom>normKLT*5)||(cosine>0&&normCom>normKLT*10));
                bool addHead = gTrk.trkType==HEAD_TRK&&dettype==HEAD_TRK&&trkCoArea>0.7;
                bool addKLT = gTrk.trkType==KLT_TRK&&dettype==KLT_TRK
                        &&(trkCoArea>0.5)
                        &&(*gType)[j]!=DETECT_INVALID
                        &&isGoodUpdate;
                if(addHead||addKLT)
                {
                float insertVal=area;
                int insertPos=0;
                int insertIdx = j;
                int startInserting=false;
                do{
                  if(counter<=0)break;
                  if(!startInserting&&valvec[insertPos]<insertVal)
                      startInserting=true;
                  if(startInserting)
                  {
                      float tmpval=valvec[insertPos];
                      int tmpidx=idxvec[insertPos];
                      valvec[insertPos]=insertVal;
                      idxvec[insertPos]=insertIdx;
                      insertVal=tmpval;
                      insertIdx=tmpidx;
                  }
                  insertPos++;
                }while(insertPos<counter);
                valvec[insertPos]=insertVal;
                idxvec[insertPos]=insertIdx;
                //if(startInserting)
                counter++;
                }
                bool addHEAD2KLT=gTrk.trkType==HEAD_TRK&&dettype==KLT_TRK&&trkCoArea>0.7;
                if(addHEAD2KLT)
                {
                    //int* headstas = gTrk.getCur_(gTrk.headStats->cpu_ptr(),CC_STAT_TOTAL);
                    float insertVal=trkBox.top;
                    int insertPos=0;
                    int insertIdx = i;
                    int startInserting=false;
                    float* jvalvec = scoreNew->cpuAt(j*nFeatures);
                    int* jidxvec = rankingNew->cpuAt(j*nFeatures);
                    int& jcounter=(*rankCountNew)[j];
                    do{
                      if(jcounter<=0)break;
                      if(!startInserting&&jvalvec[insertPos]<insertVal)
                          startInserting=true;
                      if(startInserting)
                      {
                          float tmpval=jvalvec[insertPos];
                          int tmpidx=jidxvec[insertPos];
                          jvalvec[insertPos]=insertVal;
                          jidxvec[insertPos]=insertIdx;
                          insertVal=tmpval;
                          insertIdx=tmpidx;
                      }
                      insertPos++;
                    }while(insertPos<jcounter);
                    jvalvec[insertPos]=insertVal;
                    jidxvec[insertPos]=insertIdx;
                    //if(startInserting)
                    jcounter++;
                }
            }
        }
        }
    }
    /*
    for(int i=1;i<groups->kltGroupNum;i++)
    {
        int count = (*rankCountNew)[i];
        BBox detBox=(*groups->bBox)[i];
        int goodIdx=-1;
        if(count>1)
        {
            for(int j=0;j<count;j++)
            {
                int checkIdx = (*rankingNew)[i*nFeatures+j];
                BBox jBox=*groupsTrk->getCurBBox(checkIdx);
                float shapeDiffw=abs((jBox.right-jBox.left)-(detBox.right-detBox.left))/float((jBox.right-jBox.left)+(detBox.right-detBox.left));
                float shapeDiffh=abs((jBox.bottom-jBox.top)-(detBox.bottom-detBox.top))/float((jBox.bottom-jBox.top)+(detBox.bottom-detBox.top));
                (*shapevec)[checkIdx]=shapeDiffw;
                if(shapeDiffh<0.07&&shapeDiffw<0.07&&(*rankingOld)[checkIdx*nFeatures]==i)
                {
                    goodIdx=j;
                    break;
                }
            }
            if(goodIdx<0)(*gType)[i]=DETECT_DELAY_MERGE;
            else (*gType)[i]=DETECT_GOOD;
            for(int j=0;j<count;j++)
            {
                int checkIdx=(*rankingNew)[i*nFeatures+j];
                if(j!=goodIdx)
                {
                    (*updateType)[checkIdx]=1;
                    bool startdeleting=false;
                    for(int k=0;k<(*rankCountOld)[checkIdx];k++)
                    {
                        if((*rankingOld)[checkIdx*nFeatures+k]==i)
                        {
                            startdeleting=true;
                        }
                        if(startdeleting)
                        {
                           (*rankingOld)[checkIdx*nFeatures+k]=(*rankingOld)[checkIdx*nFeatures+k+1];
                           (*scoreOld)[checkIdx*nFeatures+k]=(*scoreOld)[checkIdx*nFeatures+k+1];
                        }
                    }
                    if(startdeleting)
                        (*rankCountOld)[checkIdx]--;
                }
                else if(j==goodIdx&&j>=0)
                {
                    (*rankCountOld)[checkIdx*nFeatures]=1;
                    (*rankingOld)[checkIdx*nFeatures]=i;
                }
            }
        }
    }
    */
    updateType->toZeroH();
    //shapevec->toZeroH();
    for(int i=0;i<groupsTrk->numGroup;i++)
        for(int j=0;j<groupsTrk->numGroup;j++)
        {
            int iType = groupsTrk->getPtr(i)->trkType;
            int jType = groupsTrk->getPtr(j)->trkType;
            int icount = (*rankCountOld)[i];
            int jcount = (*rankCountOld)[j];
            if(i<j&&iType==jType&&icount&&jcount&&(*groupsTrk->vacancy)[i]&&(*groupsTrk->vacancy)[j])
            {
                int iIdx=(*rankingOld)[i*nFeatures];
                int jIdx=(*rankingOld)[j*nFeatures];
                if(iIdx==jIdx)
                {
                        int dupIdx=iIdx;
                        int dupType =(dupIdx>=groups->kltGroupNum);
                        if(dupType==iType)
                        {
                            if(dupType==KLT_TRK)
                            {

                                float iCoArea=(*scoreOld)[i*nFeatures];
                                float jCoArea=(*scoreOld)[j*nFeatures];
                                float iArea=groupsTrk->getCurArea(i);
                                float jArea=groupsTrk->getCurArea(j);
                                GroupTrack* iTrk = groupsTrk->getPtr(i),* jTrk=groupsTrk->getPtr(j);
                                float iCo= iCoArea/iArea;
                                float jCo= jCoArea/jArea;
                                if(iCo>0.7&&jCo>0.7)
                                {
                                    BBox detBox = (*groups->bBox)[dupIdx];
                                    BBox iBox = *groupsTrk->getCurBBox(i);
                                    BBox jBox = *groupsTrk->getCurBBox(j);
                                    float ishapeW=abs((iBox.right-iBox.left)-(detBox.right-detBox.left))/float((iBox.right-iBox.left)+(detBox.right-detBox.left))*2;
                                    float ishapeH=abs((iBox.bottom-iBox.top)-(detBox.bottom-detBox.top))/float((iBox.bottom-iBox.top)+(detBox.bottom-detBox.top))*2;
                                    float jshapeW=abs((jBox.right-jBox.left)-(detBox.right-detBox.left))/float((jBox.right-jBox.left)+(detBox.right-detBox.left))*2;
                                    float jshapeH=abs((jBox.bottom-jBox.top)-(detBox.bottom-detBox.top))/float((jBox.bottom-jBox.top)+(detBox.bottom-detBox.top))*2;
                                    bool shapeCompare=ishapeW*ishapeH>jshapeW*jshapeH;
//                                    std::cout<<"("<<dupIdx<<")"<<i<<":"<<ishapeW<<"|"<<j<<":"<<jshapeW<<std::endl;
//                                    shapevec[i]=ishapeW;
//                                    shapevec[j]=jshapeW;
//                                    float2 icom = *iTrk->getCurCom();
//                                    float2 jcom = *jTrk->getCurCom();
//                                    float2 detCom = (*groups->com)[dupIdx];
//                                    float2 comUpdate = {detCom.x-icom.x,detCom.y-icom.y};
//                                    float2 kltUpdate = (*kltUpdateVec)[i]
                                    if(shapeCompare&&jshapeW<0.3&&jshapeH<0.3)
                                    {
                                        //match j
                                        if((*rankCountOld)[i]>0)
                                        {
                                        for(int k=0;k<(*rankCountOld)[i]-1;k++)
                                        {
                                            (*scoreOld)[i*nFeatures+k]=(*scoreOld)[i*nFeatures+k+1];
                                            (*rankingOld)[i*nFeatures+k]=(*rankingOld)[i*nFeatures+k+1];
                                        }
                                        (*rankCountOld)[i]=(*rankCountOld)[i]-1;
                                        }
                                        (*gType)[dupIdx]=DETECT_FIRM;
                                        (*updateType)[j]=0;
                                    }
                                    else if(!shapeCompare&&ishapeW<0.3&&ishapeH<0.3)
                                    {
                                        //match i
                                        if((*rankCountOld)[j]>0)
                                        {
                                        for(int k=0;k<(*rankCountOld)[j]-1;k++)
                                        {
                                            (*scoreOld)[j*nFeatures+k]=(*scoreOld)[j*nFeatures+k+1];
                                            (*rankingOld)[j*nFeatures+k]=(*rankingOld)[j*nFeatures+k+1];
                                        }
                                        (*rankCountOld)[j]=(*rankCountOld)[j]-1;
                                        }
                                         (*gType)[dupIdx]=DETECT_FIRM;
                                        (*updateType)[i]=0;
                                    }
                                    else if((*gType)[dupIdx]!=DETECT_FIRM)
                                    {
                                        (*gType)[dupIdx]=DETECT_DELAY_MERGE;
                                        if((*rankCountOld)[j]>0)
                                        {
                                        for(int k=0;k<(*rankCountOld)[j]-1;k++)
                                        {
                                            (*scoreOld)[j*nFeatures+k]=(*scoreOld)[j*nFeatures+k+1];
                                            (*rankingOld)[j*nFeatures+k]=(*rankingOld)[j*nFeatures+k+1];
                                        }
                                        (*rankCountOld)[j]=(*rankCountOld)[j]-1;
                                        }
                                        if((*rankCountOld)[i]>0)
                                        {
                                        for(int k=0;k<(*rankCountOld)[i]-1;k++)
                                        {
                                            (*scoreOld)[i*nFeatures+k]=(*scoreOld)[i*nFeatures+k+1];
                                            (*rankingOld)[i*nFeatures+k]=(*rankingOld)[i*nFeatures+k+1];
                                        }
                                        (*rankCountOld)[i]=(*rankCountOld)[i]-1;
                                        }
                                        //(*updateType)[i]=1;
                                        //(*updateType)[j]=1;
                                    }

                                }

                            }
                            else
                            {
                                (*gType)[dupIdx]=DETECT_DELAY_MERGE;
                                (*updateType)[i]=1;
                                (*updateType)[j]=1;
                            }

                        }
                }
            }
        }
    float duration = ( std::clock() - start ) / (float) CLOCKS_PER_SEC;
    debuggingFile<<"match Time:"<<duration<<std::endl;

}
__global__ void groupsTrkPtsCount(GroupTrack* trkvec,int* vacancy,TracksInfo trkInfo,int *trkptscount)
{
    const int pIdx = threadIdx.x;
    const int gIdx = blockIdx.x;
    GroupTrack groups = trkvec[gIdx];
    int nFeatures=groups.trkPtsNum;
    const BBox gBBox= *groups.getCur_(groups.bBoxPtr);
    FeatPts curPts=trkInfo.curTrkptr[pIdx];
    float2 curVelo=trkInfo.curVeloPtr[pIdx];
    trkptscount[gIdx]=0;
    if(vacancy[gIdx]&&trkInfo.lenVec[pIdx]>0)
    {
        __shared__ int nPtsBuff[1];
        nPtsBuff[0]=0;
        __syncthreads();
        if(curPts.x>gBBox.left&&curPts.x<gBBox.right&&curPts.y>gBBox.top&&curPts.y<gBBox.bottom)
        {
            atomicAdd(nPtsBuff,1);
        }
        __syncthreads();
        trkptscount[gIdx]=nPtsBuff[0];
    }
}
void CrowdTracker::GroupTrkCorrelate()
{
    int correlateLen=11;
    for(int i=0;i<groupsTrk->numGroup;i++)
    {
        GroupTrack* iTrk=groupsTrk->getPtr(i);
        if(iTrk->trkType==KLT_TRK&&iTrk->len>correlateLen)
        {
        for(int j=0;j<groupsTrk->numGroup;j++)
        {
            GroupTrack* jTrk=groupsTrk->getPtr(j);
            bool linked = (std::find(gTrkCorr[i].begin(), gTrkCorr[i].end(), j) != gTrkCorr[i].end());
            if(jTrk->trkType==HEAD_TRK&&jTrk->len>correlateLen&&!linked)
            {
                BBox iBox=*iTrk->getCurBBox();
                BBox jBox=*jTrk->getCurBBox();
                int iArea=(iBox.right-iBox.left)*(iBox.bottom-iBox.top);
                int jArea=(jBox.right-jBox.left)*(jBox.bottom-jBox.top);
                int minw = min((iBox.right-jBox.left),(jBox.right-iBox.left));
                int minh = min((iBox.bottom-jBox.top),(jBox.bottom-iBox.top));
                UperLowerBound(minw,0,1920);
                UperLowerBound(minh,0,1080);
                int area = minw*minh;
                float iCoArea=float(area)/float(iArea);
                float jCoArea=float(area)/float(jArea);
                bool isCorelated=true;
                if(iCoArea>0.1||jCoArea>0.1)
                {
                    for(int k=0;k<correlateLen;k++)
                    {
                        BBox iBox=*iTrk->getPtr_(iTrk->bBox->cpu_ptr(),k);
                        BBox jBox=*jTrk->getPtr_(jTrk->bBox->cpu_ptr(),k);
                        int iArea=(iBox.right-iBox.left)*(iBox.bottom-iBox.top);
                        int jArea=(jBox.right-jBox.left)*(jBox.bottom-jBox.top);
                        int minw = min((iBox.right-jBox.left),(jBox.right-iBox.left));
                        int minh = min((iBox.bottom-jBox.top),(jBox.bottom-iBox.top));
                        UperLowerBound(minw,0,1920);
                        UperLowerBound(minh,0,1080);
                        int area = minw*minh;
                        float iCoArea=float(area)/float(iArea);
                       float jCoArea=float(area)/float(jArea);
                       if(iCoArea>0.7||jCoArea>0.7)
                       {
                       }
                       else
                       {
                           isCorelated=false;
                           break;
                       }
                    }
                }
                else
                {
                        isCorelated=false;
                }
                if(isCorelated)
                {
                    gTrkCorr[i].push_back(j);
                    gTrkCorr[j].push_back(i);
                    isLinked[j]=1;
                }
                else
                {
                    gTrkCorr[i].erase(std::remove(gTrkCorr[i].begin(),gTrkCorr[i].end(),j),gTrkCorr[i].end());
                    gTrkCorr[j].erase(std::remove(gTrkCorr[j].begin(),gTrkCorr[j].end(),i),gTrkCorr[j].end());
                }
            }//>len
        }//loop head TRk
        }//>len

    }//loop KLT Trk

    for(int i=0;i<groupsTrk->numGroup;i++)
    {
        GroupTrack* iTrk=groupsTrk->getPtr(i);
        if(iTrk->len<correlateLen)
        {
            gTrkCorr[i].clear();
            for(int j=0;j<groupsTrk->numGroup;j++)
            {
                gTrkCorr[j].erase(std::remove(gTrkCorr[j].begin(),gTrkCorr[j].end(),i),gTrkCorr[j].end());
            }
        }
    }
    /*
    std::fill(mergeTable.begin(),mergeTable.end(),0);
    mergeIdx.clear();
    mergeBox.clear();
    for(int i=0;i<groupsTrk->numGroup;i++)
    {

        GroupTrack* iTrk=groupsTrk->getPtr(i);
        if(iTrk->trkType==KLT_TRK&&iTrk->len>correlateLen)
        {
            if(gTrkCorr[i].size()>0)
            {
                mergeTable[i]=1;
                BBox box=*iTrk->getCurBBox();
                mergeIdx.push_back(std::vector<int>());
                for(int j=0;j<gTrkCorr[i].size();j++)
                {
                    int headIdx=gTrkCorr[i][j];
                    BBox jBox = *groupsTrk->getCurBBox(headIdx);
                    box.left=std::min(box.left,jBox.left);
                    box.top=std::min(box.top,jBox.top);
                    box.right=std::max(box.right,jBox.right);
                    box.bottom=std::max(box.bottom,jBox.bottom);
                    mergeIdx[mergeIdx.size()-1].push_back(headIdx);
                    mergeTable[headIdx]=1;
                }
                mergeBox.push_back(box);
            }
        }
        else
        {

        }
    }
    */
    //gTrkNb.setTo(0);

    int gTrkMinLen=5;
    int checkLen=1;
    for(int i=0;i<groupsTrk->numGroup;i++)
    {
        GroupTrack* iTrk=groupsTrk->getPtr(i);
        for(int j=i+1;j<groupsTrk->numGroup;j++)
        {
            GroupTrack* jTrk=groupsTrk->getPtr(j);
            if((*groupsTrk->vacancy)[i]&&(*groupsTrk->vacancy)[j]&&iTrk->len>checkLen&&jTrk->len>checkLen)
            {
                bool isNb=true;

                for(int k=0;k<checkLen;k++)
                {
                    BBox iBox= *iTrk->getPtr_(iTrk->bBox->cpu_ptr(),k);
                    BBox jBox= *jTrk->getPtr_(jTrk->bBox->cpu_ptr(),k);
                    float iArea = (iBox.right-iBox.left)* (iBox.bottom-iBox.top);
                    float jArea = (jBox.right-jBox.left)* (jBox.bottom-jBox.top);
                    int minw = min((iBox.right-jBox.left),(jBox.right-iBox.left));
                    int minh = min((iBox.bottom-jBox.top),(jBox.bottom-iBox.top));
                    UperLowerBound(minw,0,1920);
                    UperLowerBound(minh,0,1080);
                    float area = minw*minh;
                    float iCo=float(area)/float(iArea);
                    float jCo=float(area)/float(jArea);
                    //float2 iVelo = iTrk->getPtr_(iTrk->velo->cpu_ptr(),k);
                    //float2 iVelo = iTrk->getPtr_(iTrk->velo->cpu_ptr(),k);
                    float2 iVelo=*iTrk->getCur_(iTrk->velo->cpu_ptr());
                    float2 jVelo=*jTrk->getCur_(jTrk->velo->cpu_ptr());
                    float inorm = sqrt(iVelo.x*iVelo.x+iVelo.y*iVelo.y)+0.001;
                    float jnorm = sqrt(jVelo.x*jVelo.x+jVelo.y*jVelo.y)+0.001;
                    float cos = (iVelo.x*jVelo.x+iVelo.y*jVelo.y)/inorm*jnorm;
                    float veloDiff=abs(inorm-jnorm)/(inorm+jnorm);
                    //TODO
                    if((iCo>0.5||jCo>0.5)&&cos>0.5&&veloDiff<0.3)
                    {
                    }
                    else
                    {
                        isNb=false;
                        break;
                    }
                }
//                float2 iVelo=(*kltUpdateVec)[i];
//                float2 jVelo=(*kltUpdateVec)[j];
//                float2 veloDiff={iVelo.x-jVelo.x,iVelo.y-jVelo.y};
//                float veloDiffNorm=sqrt(veloDiff.x*veloDiff.x+veloDiff.y*veloDiff.y);
//                float veloFacto=1/veloDiffNorm;
                if(isNb)
                {

                    gTrkNb[i*maxGroupTrk+j]+=1;
                    gTrkNb[j*maxGroupTrk+i]+=1;
                }
                else
                {
                    gTrkNb[i*maxGroupTrk+j]/=2;
                    gTrkNb[j*maxGroupTrk+i]/=2;
                }
            }
        }
    }

    memset(bfsearchitems.data(),0,maxGroupTrk*sizeof(int));
    memcpy(bfsearchitems.data(),groupsTrk->vacancy->cpu_ptr(),maxGroupTrk*sizeof(int));
    for(int i=0;i<nFeatures;i++)
    {
        GroupTrack* trkptr=groupsTrk->getPtr(i);
        bfsearchitems[i]=i*((bfsearchitems[i]>0)&&(trkptr->updateCount/trkptr->len<0.05));
    }
    memset(gTrkLabel,0,maxGroupTrk*sizeof(int));
    memset(gTrkbbNum,0,maxGroupTrk*sizeof(int));
    memset(idxmap,0,nFeatures*sizeof(int));
    memset(tmpn,0,nFeatures*sizeof(int));
    int* h_label=gTrkLabel;
    int* h_gcount=gTrkbbNum;
    int* h_neighbor=gTrkNb;
    int idx=0;
    int total=0;
    bool unset=true;
    int gcount=0;
    int minCount=0;
    for(int i=0;i<maxGroupTrk;i++)
    {
        if(unset&&bfsearchitems[i]>0){idx=i;unset=false;}
        total+=(bfsearchitems[i]>0);
        if(!unset)
        {
            tmpn[i]+=(h_neighbor[idx*maxGroupTrk+i]>minCount);
        }
    }
    bfsearchitems[idx]=0;
    total--;
    std::cout<<"gTrk total BFS:"<<total<<std::endl;
    int gTrkCurK=1;
    gTrkGroupN=0;
    gcount++;
    while(total>0)
    {
        int ii=0;
        for(idx=0;idx<maxGroupTrk;idx++)
        {
            if(!ii)ii=idx*(bfsearchitems[idx]>0);
            if(bfsearchitems[idx]&&tmpn[idx])
            {
//                int nc=0,nnc=0;
//                float nscore=0;
//                for(int i=0;i<maxGroupTrk;i++)
//                {
//                    if(h_neighbor[idx*maxGroupTrk+i])
//                    {
//                        nc+=(h_neighbor[idx*maxGroupTrk+i]>0);
//                        nnc+=(tmpn[i]>0);
//                    }
//                }
//                if(nnc>nc*0.4+1)
                {
                    gcount++;
                    h_label[idx]=gTrkCurK;
                    for(int i=0;i<maxGroupTrk;i++)
                    {
                        tmpn[i]+=(h_neighbor[idx*maxGroupTrk+i]>minCount);
                    }
                    bfsearchitems[idx]=0;
                    total--;
                    if(ii==idx)ii=0;
                 }
            }
        }
        if(gcount>0)
        {
            h_gcount[gTrkCurK]+=gcount;
            gcount=0;
        }
        else if(total>0)
        {
            if(h_gcount[gTrkCurK]>minGSize)
            {
                gTrkGroupN++;
                idxmap[gTrkCurK]=gTrkGroupN;
            }
            gTrkCurK++;
            gcount=0;
            memset(tmpn,0,nFeatures*sizeof(int));
            idx=ii;
            gcount++;
            h_label[idx]=gTrkCurK;
            for(int i=0;i<maxGroupTrk;i++)
            {
                tmpn[i]+=(h_neighbor[idx*maxGroupTrk+i]>minCount);
            }
            bfsearchitems[idx]=0;
            total--;
        }
    }
    for(int i=0;i<maxGroupTrk;i++)
    {
        h_label[i]=idxmap[h_label[i]];
    }
    memset(gTrkbbNum,0,maxGroupTrk*sizeof(int));
    for(int i=0;i<gTrkGroupN;i++)
    {
        mergeIdx[i].resize(0);
        mergeIdx[i].resize(maxGroupTrk,0);
        mergeBox[i].left=1920;
        mergeBox[i].top=1080;
        mergeBox[i].right=0;
        mergeBox[i].bottom=0;

    }
    std::cout<<"gTrkGroupN:"<<gTrkGroupN<<std::endl;
    for(int i=0;i<groupsTrk->numGroup;i++)
    {
        int gIdx=h_label[i];
        if(gIdx>0)
        {
            mergeIdx[gIdx].push_back(i);
            BBox bb = *groupsTrk->getCurBBox(i);
            mergeBox[gIdx].left=min(mergeBox[gIdx].left,bb.left);
            mergeBox[gIdx].top=min(mergeBox[gIdx].top,bb.top);
            mergeBox[gIdx].right=max(mergeBox[gIdx].right,bb.right);
            mergeBox[gIdx].bottom=max(mergeBox[gIdx].bottom,bb.bottom);
            int& counter=gTrkbbNum[gIdx];
            int insertPos=0;
            int insertIdx = i;
            int startInserting=false;
            do{
              if(counter<=0)break;
              if(!startInserting&&(*groupsTrk)[mergeIdx[gIdx][insertPos]].len<(*groupsTrk)[insertIdx].len)
                  startInserting=true;
              if(startInserting)
              {
                  int tmpidx=mergeIdx[gIdx][insertPos];
                  mergeIdx[gIdx][insertPos]=insertIdx;
                  insertIdx=tmpidx;
              }
              insertPos++;
            }while(insertPos<counter);
            mergeIdx[gIdx][insertPos]=insertIdx;
            //if(startInserting)
            counter++;
        }
    }
    groupsTrkPtsCount<<<groupsTrk->numGroup,nFeatures>>>(groupsTrk->groupTracks->gpu_ptr(),groupsTrk->vacancy->gpu_ptr(),trkInfo,trkptscount->gpu_ptr());
    trkptscount->SyncD2H();
}
__global__ void applyFgMask(unsigned char* d_mask, unsigned char* d_fmask)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    int w = d_framewidth[0], h = d_frameheight[0];
    int offset = y*w + x;
    /*
    const int range = 5;
    if (x < w - range && y < h - range && x>range && y>range)
    {
        bool isFG = false;
        for (int i = -range; i<range; i++)
            for (int j = -range; j <range; j++)
            {
                isFG = isFG||d_fmask[(y + range)*w + x + range];
            }
        d_mask[offset] = d_mask[offset] && isFG;
    }
    */
    if (x < w && y < h)
    {
        d_mask[offset] = d_mask[offset] && d_fmask[offset];
    }
}
inline void buildPolygon(float2* pts,int& ptsCount,float2* polygon,int& polyCount)
{
    cvxPnt* P=(cvxPnt*)pts;
    cvxPnt* H=(cvxPnt*)polygon;
    int n = ptsCount, k = 0;
    // Sort points lexicographically
    std::sort(P,P+ptsCount);
    // Build lower hull
    for (int i = 0; i < n; ++i) {
        while (k >= 2 && cross_(H[k-2], H[k-1], P[i]) <= 0) k--;
        H[k++]=P[i];
    }

    // Build upper hull
    for (int i = n-2, t = k+1; i >= 0; i--) {
        while (k >= t && cross_(H[k-2], H[k-1], P[i]) <= 0) k--;
        H[k++]=P[i];
    }

    polyCount=k;
}

__global__ void groupsPtsCorr(Groups groups,TracksInfo trkInfo,int* d_neighbor,float* persmap,float* d_cosine)
{
    const int pIdx = threadIdx.x;
    const int gIdx = blockIdx.x;
    int nFeatures=groups.trkPtsNum;
    const BBox gBBox= groups.bBoxPtr[gIdx];
    const int oriNumPts = groups.ptsNumPtr[gIdx];
    int* idxPtr = groups.trkPtsIdxPtr+gIdx*nFeatures;
    FeatPts curPts=trkInfo.curTrkptr[pIdx];
    float2 curVelo=trkInfo.curVeloPtr[pIdx];
    if(oriNumPts==0&&trkInfo.lenVec[pIdx]>0)
    {
        __shared__ int nPtsBuff[1];
        __shared__ float velo[2];
        nPtsBuff[0]=0;
        velo[0]=0,velo[1]=0;
        __syncthreads();
        if(curPts.x>gBBox.left&&curPts.x<gBBox.right&&curPts.y>gBBox.top&&curPts.y<gBBox.bottom)
        {
            int pos = atomicAdd(nPtsBuff,1);
            idxPtr[pos]=pIdx;
            atomicAdd(velo,curVelo.x);
            atomicAdd((velo+1),curVelo.y);
        }
        __syncthreads();
        groups.ptsNumPtr[gIdx]=nPtsBuff[0];
        if(nPtsBuff[0]>0)
        {
            groups.veloPtr[gIdx].x=velo[0]/nPtsBuff[0];
            groups.veloPtr[gIdx].y=velo[1]/nPtsBuff[0];
        }
        else
        {
            groups.veloPtr[gIdx].x=0;
            groups.veloPtr[gIdx].y=0;
        }
    }
    __shared__ int nbSum[1];
    nbSum[0]=0;
    __syncthreads();
    int curNumPts = groups.ptsNumPtr[gIdx];
    for(int i=0;i<curNumPts;i++)
    {
        if(pIdx>i&&pIdx<curNumPts)
        {
            float vrx = trkInfo.curVeloPtr[i].x, vry = trkInfo.curVeloPtr[i].y
                , vcx = trkInfo.curVeloPtr[pIdx].x, vcy = trkInfo.curVeloPtr[pIdx].y;
            float normr = trkInfo.curSpdPtr[i], normc = trkInfo.curSpdPtr[pIdx];
            float veloCo = abs(normr - normc) / (normr + normc);
            float cosine = (vrx*vcx + vry*vcy) / normr / normc;
            //atomicAdd(nbSum,d_neighbor[i*nFeatures+pIdx]);
            atomicAdd(nbSum,cosine/(veloCo+0.1));
        }
    }
    __syncthreads();
    float persval=persmap[gBBox.top*d_framewidth[0]+gBBox.left];
    float nbAvg=0;
    if(curNumPts>0)
    {
        nbAvg=float(nbSum[0])/curNumPts/(curNumPts+1)*2;
    }
    groups.ptsCorrPtr[gIdx]=nbAvg;
}
void CrowdTracker::addHeadBox()
{
    groups->headStats->toZeroH();
    for(int i=1;i<nHeadLabel;i++)
    {
        double2 lfCom=(*mcentroidsv)[i];
        int* headstats=mstatsv->cpuAt(i);
        BBox headBox={headstats[CC_STAT_LEFT],headstats[CC_STAT_TOP]
                      ,headstats[CC_STAT_LEFT]+headstats[CC_STAT_WIDTH]
                     ,headstats[CC_STAT_TOP]+headstats[CC_STAT_HEIGHT]};

        int2 headCom = {lfCom.x+0.5,lfCom.y+0.5};
        float persVal = (*persMap)[headCom.y*frame_width+headCom.x];
        int bboxw=persVal/2,bboxh=persVal*2;
        int minlen = persVal/8;
        if((*roimask)[headCom.y*frame_width+headCom.x]>0&&headstats[CC_STAT_WIDTH]>minlen&&headstats[CC_STAT_HEIGHT]>minlen)
        {
        UperLowerBound(bboxw,headstats[CC_STAT_WIDTH]/2,1920);
        bboxh=bboxh+headstats[CC_STAT_HEIGHT];
        BBox bodyBBox = {headCom.x-bboxw,headCom.y-bboxh*0.09,headCom.x+bboxw,headCom.y+bboxh*0.91};
        UperLowerBound(bodyBBox.left,0,frame_width-10);
        UperLowerBound(bodyBBox.top,0,frame_height-10);
        UperLowerBound(bodyBBox.left,0,frame_width-10);
        UperLowerBound(bodyBBox.bottom,0,frame_height-10);
        bodyBBox.top=min(headBox.top,bodyBBox.top);
        int cnnArea = (bodyBBox.right-bodyBBox.left)*(bodyBBox.bottom-bodyBBox.top);
        /*
        bool isAdding=false;
        for(int j=1;j<groups->numGroups;j++)
        {
            BBox kltBBox = (*groups->bBox)[j];
            int minw = min((headBBox.right-kltBBox.left),(kltBBox.right-headBBox.left));
            int minh = min((headBBox.bottom-kltBBox.top),(kltBBox.bottom-headBBox.top));
            UperLowerBound(minw,0,1920);
            UperLowerBound(minh,0,1080);
            int coArea = minw*minh;
            if(float(coArea)/float(cnnArea)>0.5)
            {
                isAdding=true;
                break;
            }
        }
        */
        float2 manCom = {headCom.x,headCom.y+bboxh*0.4};
        int addidx = groups->numGroups;
        float2 tmpVelo = {0,0};
        if(addidx<groups->maxNumGroup)
        {
            memcpy(groups->headStats->cpuAt(addidx),headstats,CC_STAT_MAX*sizeof(int));
            memcpy(groups->headStats->cpuAt(addidx)+CC_STAT_MAX,&headCom,sizeof(int2));
            (*groups->com)[addidx]=manCom;
            (*groups->velo)[addidx]=tmpVelo;
            (*groups->bBox)[addidx]=bodyBBox;
            (*groups->ptsNum)[addidx]=0;
            (*groups->area)[addidx]=cnnArea;
            (*gHeadCount)[addidx]=1;
            (*gHeadIdxMat)[addidx*nFeatures]=i;
            //(*gType)[addidx]=1;
            groups->numGroups=groups->numGroups+(addidx==groups->numGroups);

        }
        }
    }
}
__global__ void reinitKLTGroup(Groups groups,TracksInfo trkInfo)
{
    const int pIdx = threadIdx.x;
    const int gIdx = blockIdx.x;
    int nFeatures=groups.trkPtsNum;
    const BBox gBBox= groups.bBoxPtr[gIdx];
    const int oriNumPts = groups.ptsNumPtr[gIdx];
    int* idxPtr = groups.trkPtsIdxPtr+gIdx*nFeatures;
    FeatPts curPts=trkInfo.curTrkptr[pIdx];
    float2 curVelo=trkInfo.curVeloPtr[pIdx];
    if(oriNumPts==0&&trkInfo.lenVec[pIdx]>0)
    {
        __shared__ int nPtsBuff[1];
        __shared__ float velo[2],com[2];
        nPtsBuff[0]=0;
        velo[0]=0,velo[1]=0;
        com[0]=0,com[1]=0;
        __syncthreads();
        if(curPts.x>gBBox.left&&curPts.x<gBBox.right&&curPts.y>gBBox.top&&curPts.y<gBBox.bottom)
        {
            int pos = atomicAdd(nPtsBuff,1);
            idxPtr[pos]=pIdx;
            atomicAdd(velo,curVelo.x);
            atomicAdd((velo+1),curVelo.y);
            //atomicAdd(com,curPts.x);
            //atomicAdd((com+1),curPts.y);
        }
        __syncthreads();
        groups.ptsNumPtr[gIdx]=nPtsBuff[0];
        if(nPtsBuff[0]>0)
        {
            groups.veloPtr[gIdx].x=velo[0]/nPtsBuff[0];
            groups.veloPtr[gIdx].y=velo[1]/nPtsBuff[0];
            //groups.comPtr[gIdx].x=com[0]/nPtsBuff[0];
            //groups.comPtr[gIdx].y=com[1]/nPtsBuff[0];
        }
        else
        {
            groups.veloPtr[gIdx].x=0;
            groups.veloPtr[gIdx].y=0;
        }
    }
}
void CrowdTracker::splitGroup()
{
    rankCountNew->toZeroH();
    rankingNew->toZeroH();
    scoreNew->toZeroH();
    for(int i=1;i<groups->kltGroupNum;i++)
    {
        int* idxvec=rankingNew->cpuAt(i*nFeatures);
        float* valvec=scoreNew->cpuAt(i*nFeatures);
        int& counter=(*rankCountNew)[i];
        BBox newBox = (*groups->bBox)[i];
        //float newArea = (*groups->area)[i];
        for(int j=0;j<groupsTrk->numGroup;j++)
        {
            GroupTrack* trkptr = groupsTrk->getPtr(j);
            if((*groupsTrk->vacancy)[j]&&trkptr->trkType==HEAD_TRK&&trkptr->len>11)
            {
                BBoxF trkBox=(*kltUpdateBoxVec)[j];
                float trkArea=groupsTrk->getCurArea(j);
                float minw = min((trkBox.right-newBox.left),(newBox.right-trkBox.left));
                float minh = min((trkBox.bottom-newBox.top),(newBox.bottom-trkBox.top));
                UperLowerBound(minw,0,1920);
                UperLowerBound(minh,0,1080);
                float area = minw*minh;
                float trkCoArea=float(area)/float(trkArea);
                if(trkCoArea>0.5)
                {
                    float insertVal=trkBox.top;
                    int insertPos=0;
                    int insertIdx = j;
                    int startInserting=false;
                    do{
                      if(counter<=0)break;
                      if(!startInserting&&valvec[insertPos]<insertVal)
                          startInserting=true;
                      if(startInserting)
                      {
                          float tmpval=valvec[insertPos];
                          int tmpidx=idxvec[insertPos];
                          valvec[insertPos]=insertVal;
                          idxvec[insertPos]=insertIdx;
                          insertVal=tmpval;
                          insertIdx=tmpidx;
                      }
                      insertPos++;
                    }while(insertPos<counter);
                    valvec[insertPos]=insertVal;
                    idxvec[insertPos]=insertIdx;
                    counter++;
                }
            }
        }
    }

    shapevec->toZeroH();
    for(int i=1;i<groups->kltGroupNum;i++)
    {
        int iType = KLT_TRK;
        int& counter=(*rankCountNew)[i];
        BBox iBox = (*groups->bBox)[i];
        float iw=(iBox.right-iBox.left),ih=(iBox.bottom-iBox.top);
        float iArea = (iBox.bottom-iBox.top)*(iBox.right-iBox.left)+0.01;
        bool isSplit=false;
        std::vector<int> idxvec;
        if(counter>1)
        {

            for(int j=0;j<counter;j++)
            {
                int jIdx =(*rankingNew)[i*nFeatures+j];
                BBoxF jBox = (*kltUpdateBoxVec)[jIdx];
                float jw=(jBox.right-jBox.left),jh=(jBox.bottom-jBox.top);
                float jArea = (jBox.bottom-jBox.top)*(jBox.right-jBox.left)+0.01;
                bool validSplit=true;
                for(int k=0;k<idxvec.size();k++)
                {
                    //TODO find where to split
                    int kIdx =idxvec[k];
                    BBoxF kBox =  (*kltUpdateBoxVec)[kIdx];
                    float kw=(kBox.right-kBox.left),kh=(kBox.bottom-kBox.top);
                    float kArea = (kBox.bottom-kBox.top)*(kBox.right-kBox.left)+0.01;
                    float minw = min((kBox.right-jBox.left),(jBox.right-kBox.left));
                    float minh = min((kBox.bottom-jBox.top),(jBox.bottom-kBox.top));
                    UperLowerBound(minw,0,1920);
                    UperLowerBound(minh,0,1080);
                    float area = minw*minh;
                    float kCo = area/kArea;
                    float jCo = area/jArea;
                    bool splitleft = kBox.left>jBox.right-kw*0.1&&kBox.left<jBox.right+kw*0.1;
                    bool splitright = kBox.left>jBox.right-kw*0.1&&kBox.left<jBox.right+kw*0.1;
                    if(kCo<0.1)
                    {

                    }
                    else
                    {
                            validSplit=false;
                            break;
                    }

                }
                if(validSplit)
                {
                    idxvec.push_back(jIdx);
                }
            }
            if(idxvec.size()>1)
            {
            BBoxF splitRange={1920,1080,0,0};
            float bw=0,bh=0;
            for(int j=0;j<idxvec.size();j++)
            {
                int jIdx=idxvec[j];
                BBoxF jBox = (*kltUpdateBoxVec)[jIdx];
                splitRange.left=min(jBox.left,splitRange.left);
                splitRange.top=min(jBox.top,splitRange.top);
                splitRange.right=max(jBox.right,splitRange.right);
                splitRange.bottom=max(jBox.bottom,splitRange.bottom);
                bw+=(jBox.right-jBox.left);
                bh+=(jBox.bottom-jBox.top);
            }
            float sArea = (splitRange.bottom-splitRange.top)*(splitRange.right-splitRange.left);
            float minw = min((iBox.right-splitRange.left),(iBox.right-splitRange.left));
            float minh = min((iBox.bottom-splitRange.top),(iBox.bottom-splitRange.top));
            UperLowerBound(minw,0,1920);
            UperLowerBound(minh,0,1080);
            float area = minw*minh;
            float iCo =area/iArea;
            float sCo =area/sArea;
            (*shapevec)[i]=iCo;
            if(iCo>0.8&&bw>0.8*iw&&bh>0.8*ih)
            {
                isSplit=true;
            }
            }
        }
        if(isSplit)
        {
            for(int j=0;j<idxvec.size();j++)
            {
                int headIdx=idxvec[j];
                BBoxF headBox=(*kltUpdateBoxVec)[headIdx];
                BBox box={headBox.left+0.5,headBox.top+0.5,headBox.right+0.5,headBox.bottom+0.5};
                int addidx=j>0?groups->numGroups:i;
                (*groups->bBox)[addidx]=box;
                (*groups->com)[addidx].x=(headBox.left+headBox.right)*0.5;
                (*groups->com)[addidx].y=(headBox.top+headBox.bottom)*0.5;
                (*groups->ptsNum)[addidx]=0;
                (*groups->velo)[addidx].x=0;
                (*groups->velo)[addidx].y=0;
                groups->numGroups++;
            }
        }
    }
    groups->SyncH2D();
    reinitKLTGroup<<<groups->numGroups,nFeatures>>>(*groups, trkInfo);
    groups->ptsNum->SyncD2H();
    groups->trkPtsIdx->SyncD2H();
    groups->velo->SyncD2H();
}
void CrowdTracker::makeGroups()
{
    //renderPts();
    gType->toZeroH();
    std::cout<<"making Groups"<<std::endl;
    label->SyncH2D();
    prelabel->SyncH2D();
    groups->numGroups=groupN;
    groups->headStats->toZeroH();
    groups->headStats->toZeroD();
    if(groupN)
        makeGroupKernel<<<groupN,nFeatures>>>(label->gpu_ptr(),*groups,trkInfo);
    groups->kltGroupNum=groupN;
    groups->SyncD2H();
    for(int i=0;i<groups->numGroups;i++)
    {
        BBox newBox = (*groups->bBox)[i];
        float newArea = (*groups->area)[i];
        float persval=(*persMap)[newBox.top*frame_width+int((newBox.left+newBox.right)*0.5+0.5)];
        float minArea = persval*persval/4;
        int detectionType=DETECT_GOOD;
        if(newArea<minArea)
        {   (*gType)[i]=DETECT_INVALID;
        }
    }
    splitGroup();
    groups->kltGroupNum=groups->numGroups;
    //checkNSplit();
    addHeadBox();
    groups->SyncH2D();
    std::cout<<"numGroups:"<<groupN<<","<<groups->numGroups<<std::endl;
    //rankDetection();
    groupsPtsCorr<<<groups->numGroups,groups->numGroups>>>(*groups, trkInfo,nbCount->gpu_ptr(),persMap->gpu_ptr(),cosCo->gpu_ptr());
    groups->SyncD2H();

}
__global__ void checkDup(uchar* gType,Groups groups,int* countvec,int* idxmat)
{
    const int i=blockIdx.x;
    const int j=threadIdx.x;
    if(i<j)
    {
        if(i>=groups.kltGroupNum&&j>=groups.kltGroupNum)
        {
            BBox iBBox=groups.bBoxPtr[i];
            BBox jBBox=groups.bBoxPtr[j];
            int* iHead=groups.headStatsPtr+CC_STAT_TOTAL*i;
            int* jHead=groups.headStatsPtr+CC_STAT_TOTAL*j;
            int iHeadArea=iHead[CC_STAT_WIDTH]*iHead[CC_STAT_HEIGHT];
            int jHeadArea=jHead[CC_STAT_WIDTH]*jHead[CC_STAT_HEIGHT];
            int iArea=(iBBox.right-iBBox.left)*(iBBox.bottom-iBBox.top);
            int jArea=(jBBox.right-jBBox.left)*(jBBox.bottom-jBBox.top);
            int minw = min((iBBox.right-jBBox.left),(jBBox.right-iBBox.left));
            int minh = min((iBBox.bottom-jBBox.top),(jBBox.bottom-iBBox.top));
            int ijw=min((iBBox.right-jHead[CC_STAT_LEFT]),(jHead[CC_STAT_LEFT]+jHead[CC_STAT_WIDTH]-iBBox.left));
            int ijh=min((iBBox.bottom-jHead[CC_STAT_TOP]),(jHead[CC_STAT_TOP]+jHead[CC_STAT_HEIGHT]-iBBox.top));
            int jiw = min((iHead[CC_STAT_LEFT]+iHead[CC_STAT_WIDTH]-jBBox.left),(jBBox.right-iHead[CC_STAT_LEFT]));
            int jih = min((iHead[CC_STAT_TOP]+iHead[CC_STAT_HEIGHT]-jBBox.top),(jBBox.bottom-iHead[CC_STAT_TOP]));
            UperLowerBound(minw,0,1920);
            UperLowerBound(ijw,0,1920);
            UperLowerBound(jiw,0,1920);
            UperLowerBound(minh,0,1080);
            UperLowerBound(ijh,0,1080);
            UperLowerBound(jih,0,1080);
            int ijarea = ijw*ijh;
            int jiarea = jiw*jih;
            int area = minw*minh;
            //minwiHj = min((iBBox.right-jBBox.left),(jBBox.right-iBBox.left));
            float iCoArea=float(area)/float(iArea);
            float jCoArea=float(area)/float(jArea);
            float ijCoj=float(ijarea)/float(jHeadArea);
            float jiCoi=float(jiarea)/float(iHeadArea);
            //printf("%d,%d|",i,j);
            if(iCoArea>0.7||jCoArea>0.7)
            //if(ijCoj>0.2&&jiCoi>0.2)
            {
                if(iBBox.top<jBBox.top)
                {
                    gType[j]=DETECT_DUPLICATED;
                    //gType[i]=DETECT_GOOD;

                }
                else
                {
                    gType[i]=DETECT_DUPLICATED;
                    //gType[j]=DETECT_GOOD;

                }
                int pos = atomicAdd(countvec+i,1);
                idxmat[i*NUMTHREAD+pos]=j;
                pos = atomicAdd(countvec+j,1);
                idxmat[j*NUMTHREAD+pos]=i;
                //gType[i]=DETECT_MERGE;
//                gType[j]=DETECT_MERGE;
//                //gType[i]=DETECT_MERGE;
//                groups.bBoxPtr[i].left=min(jBBox.left,iBBox.left);
//                groups.bBoxPtr[i].top=min(jBBox.top,iBBox.top);
//                groups.bBoxPtr[i].right=max(jBBox.right,iBBox.right);
//                groups.bBoxPtr[i].bottom=max(jBBox.bottom,iBBox.bottom);
//                atomicMin(&(groups.bBoxPtr[i].left),jBBox.left);
//                atomicMin(&(groups.bBoxPtr[i].top),jBBox.top);
//                atomicMax(&(groups.bBoxPtr[i].right),jBBox.right);
//                atomicMax(&(groups.bBoxPtr[i].bottom),jBBox.bottom);
            }
        }
    }
    __syncthreads();
    if(i>0&&i<groups.kltGroupNum&&j<groups.kltGroupNum)
    {
        int* countptr=countvec+i;
        int* idxvec=idxmat+i*NUMTHREAD;
        if(!gType[j])
        {
            BBox iBBox=groups.bBoxPtr[i];
            BBox jBBox=groups.bBoxPtr[j];
            int iArea=(iBBox.right-iBBox.left)*(iBBox.bottom-iBBox.top);
            int jArea=(jBBox.right-jBBox.left)*(jBBox.bottom-jBBox.top);
            int minw = min((iBBox.right-jBBox.left),(jBBox.right-iBBox.left));
            int minh = min((iBBox.bottom-jBBox.top),(jBBox.bottom-iBBox.top));
            UperLowerBound(minw,0,1920);
            UperLowerBound(minh,0,1080);
            int area = minw*minh;
             float iCoArea=float(area)/float(iArea);
            float jCoArea=float(area)/float(jArea);
            if(jCoArea>0.5||iCoArea>0.5)
            {
                //printf("%d,%d|",area,jArea);
                int pos = atomicAdd(countptr,1);
                idxvec[pos]=j;
            }

        }
    }
}
void CrowdTracker::rankDetection()
{
    gHeadCount->toZeroD();
    gType->SyncH2D();
    checkDup<<<groups->numGroups,groups->numGroups>>>(gType->gpu_ptr(),*groups,gHeadCount->gpu_ptr(),gHeadIdxMat->gpu_ptr());
    gType->SyncD2H();
    //groups->bBox->SyncD2H();
    gHeadCount->SyncD2H();
    gHeadIdxMat->SyncD2H();
    for(int i=1;i<groups->kltGroupNum;i++)
    {
        BBox bb=(*groups->bBox)[i];
        float persval=(*persMap)[bb.top*frame_width+int((bb.left+bb.right)*0.5+0.5)];
        (*groups->shape)[i].x=(bb.right-bb.left)/persval;
        (*groups->shape)[i].y=(bb.bottom-bb.top)/persval;
        float minArea = persval*persval/4;
        if((*groups->area)[i]<minArea)
        {
            (*gType)[i]=DETECT_INVALID;
        }
//        else if((*gHeadCount)[i]>1)
//        {
//            (*gType)[i]=DETECT_SPLITED;
//        }
//        else if((*gHeadCount)[i]==1)
//        {
//            (*gType)[i]=DETECT_1V1;

//        }

        /*
        else
        {
            (*gType)[i]=2;
        }
        */
    }
}
__global__ void enlistPoints(float* framePtr,int2* listPtr)
{
    int fw = d_framewidth[0];
    int fh = d_frameheight[0];

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    int offset=y*fw+x;
    if(framePtr[offset]>0)
    {
        int posidx = atomicAdd(d_total, 1);
    }
}
__global__ void threshPersKernek(float* origin,unsigned char* binary,float* persmap,float minVal,float maxVal)
{
    int fw = d_framewidth[0];
    int fh = d_frameheight[0];

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int offset = y*fw+x;
    float val = origin[offset];
    float persVal=persmap[offset];
    float persThresh =(maxVal-minVal)/(sqrt(persVal)+3)*2+minVal;
    binary[offset]=(val>persThresh)*255;

}
void CrowdTracker::detectHead()
{

    headMapOrigin->toZeroH();
    std::cout<<"head fidx:"<<(*fidxBuff)[mididx]<<std::endl;
    sprintf(strbuff,"%08d.bin",(*fidxBuff)[mididx]+1);
    std::string filepath = headDetectPath+strbuff;
    headFileReader.open(filepath.c_str(),std::ios::in | std::ios::binary);
    cudaSafeCall(cudaGetLastError());
    if (headFileReader.is_open())
    {
        headFileReader.read((char*)headMapOrigin->cpu_ptr(), headDetectorSize*sizeof(float));
        headFileReader.close();
    }
    else
    {
        std::cout<<"file open failed"<<std::endl;
    }
    std::cout<<filepath<<std::endl;
    //while(headFileReader.is_open());
    headMapOrigin->SyncH2D();

    gpu::GpuMat small(headDetectorH,headDetectorW,CV_32F,headMapOrigin->gpu_ptr());
    gpu::GpuMat big(frame_height,frame_width,CV_32F,headMap->gpu_ptr());
    gpu::GpuMat bigBin(frame_height,frame_width,CV_32F,headBin->gpu_ptr());
    gpu::resize(small,big,Size(frame_width,frame_height));
    gpu::minMax(big,&headMin,&headMax);
    size_t freeMemSize,usedMemSize;
    cudaMemGetInfo(&freeMemSize,&usedMemSize);
    std::cout<<"free:"<<freeMemSize/1024/1024<<"|used:"<<usedMemSize/1024/1024<<std::endl;
    /*
    float threshval = (headMax-headMin)/3+headMin;
    gpu::threshold(big,bigBin,threshval,255,THRESH_BINARY);
    headBin->SyncD2H();
    */

    dim3 block(32, 32);
    dim3 grid(divUp(frame_width, 32), divUp(frame_height, 32));
    threshPersKernek<<<grid,block>>>(headMap->gpu_ptr(),headBinUchar->gpu_ptr(),persMap->gpu_ptr(),headMin,headMax);
    Mat components(frame_height,frame_width,CV_32S,labelMap->cpu_ptr());

    Mat bigBinCPU(frame_height,frame_width,CV_32F,headBin->cpu_ptr());
    Mat binUchar(frame_height,frame_width,CV_8UC1,headBinUchar->cpu_ptr());
    //bigBinCPU.convertTo(binUchar,CV_8UC1);
    Mat stats(cv::Size(CC_STAT_MAX, nFeatures),CV_32S,mstatsv->cpu_ptr());
    Mat cent(cv::Size(2, nFeatures),CV_64F,mcentroidsv->cpu_ptr());
    headBinUchar->SyncD2H();
    //Mat stats,cent;
    nHeadLabel = cv::connectedComponentsWithStats(binUchar,components,stats,cent);
    labelMap->SyncH2D();
    mstatsv->SyncH2D();
    mcentroidsv->SyncH2D();
    std::cout<<"Size:"<<nHeadLabel<<std::endl;
    unsigned char* h_clrvec=clrvec->cpu_ptr();
    for(int i=0;i<nHeadLabel;i++)
    {
        HSVtoRGB(h_clrvec+i*3,h_clrvec+i*3+1,h_clrvec+i*3+2,i/(nHeadLabel+0.01)*360,1,1);
    }
    clrvec->SyncH2D();
    std::cout<<components.size()<<std::endl;
    /*
    dim3 block(32, 32);
    dim3 grid(divUp(frame_width, 32), divUp(frame_height, 32));

    MapEnlarge<<<grid,block>>>(headDetectorW,headDetectorH,headMapOrigin->cpu_ptr(),headMap->gpu_ptr());
    */
}
