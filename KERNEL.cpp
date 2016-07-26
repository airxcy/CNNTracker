	unsigned char tmpz[27] =
	{
		0, 0, 0, 0, 0, 0, 0, 0, 0,
		1, 1, 1, 1, 1, 1, 1, 1, 1,
		2, 2, 2, 2, 2, 2, 2, 2, 2
	};
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
		result += sobelFilter[0] * volume[(zs[0] * fh + ys[0])*fw + xs[0]];
		result += sobelFilter[1] * volume[(zs[0] * fh + ys[0])*fw + xs[1]];
		result += sobelFilter[2] * volume[(zs[0] * fh + ys[0])*fw + xs[2]];
		result += sobelFilter[3] * volume[(zs[0] * fh + ys[1])*fw + xs[0]];
		result += sobelFilter[4] * volume[(zs[0] * fh + ys[1])*fw + xs[1]];
		result += sobelFilter[5] * volume[(zs[0] * fh + ys[1])*fw + xs[2]];
		result += sobelFilter[6] * volume[(zs[0] * fh + ys[2])*fw + xs[0]];
		result += sobelFilter[7] * volume[(zs[0] * fh + ys[2])*fw + xs[1]];
		result += sobelFilter[8] * volume[(zs[0] * fh + ys[2])*fw + xs[2]];
                              
		result += sobelFilter[9]] * volume[(zs[1] * fh + ys[0])*fw + xs[0]];
		result += sobelFilter[10] * volume[(zs[1] * fh + ys[0])*fw + xs[1]];
		result += sobelFilter[11] * volume[(zs[1] * fh + ys[0])*fw + xs[2]];
		result += sobelFilter[12] * volume[(zs[1] * fh + ys[1])*fw + xs[0]];
		result += sobelFilter[13] * volume[(zs[1] * fh + ys[1])*fw + xs[1]];
		result += sobelFilter[14] * volume[(zs[1] * fh + ys[1])*fw + xs[2]];
		result += sobelFilter[15] * volume[(zs[1] * fh + ys[2])*fw + xs[0]];
		result += sobelFilter[16] * volume[(zs[1] * fh + ys[2])*fw + xs[1]];
		result += sobelFilter[17] * volume[(zs[1] * fh + ys[2])*fw + xs[2]];
		                      
		result += sobelFilter[18] * volume[(zs[2] * fh + ys[0])*fw + xs[0]];
		result += sobelFilter[19] * volume[(zs[2] * fh + ys[0])*fw + xs[1]];
		result += sobelFilter[20] * volume[(zs[2] * fh + ys[0])*fw + xs[2]];
		result += sobelFilter[21] * volume[(zs[2] * fh + ys[1])*fw + xs[0]];
		result += sobelFilter[22] * volume[(zs[2] * fh + ys[1])*fw + xs[1]];
		result += sobelFilter[23] * volume[(zs[2] * fh + ys[1])*fw + xs[2]];
		result += sobelFilter[24] * volume[(zs[2] * fh + ys[2])*fw + xs[0]];
		result += sobelFilter[25] * volume[(zs[2] * fh + ys[2])*fw + xs[1]];
		result += sobelFilter[26] * volume[(zs[2] * fh + ys[2])*fw + xs[2]];
		
#define applyKernel(x,y,z) \  
result += sobelFilter[(z[0] * 183 + y[0]) * 3 + x[0]] * volume[(zs[0] * fh + ys[0])*fw + xs[0]];\
result += sobelFilter[(z[1] * 193 + y[1]) * 3 + x[1]] * volume[(zs[0] * fh + ys[0])*fw + xs[1]];\
result += sobelFilter[(z[2] * 203 + y[2]) * 3 + x[2]] * volume[(zs[0] * fh + ys[0])*fw + xs[2]];\
result += sobelFilter[(z[3] * 213 + y[3]) * 3 + x[3]] * volume[(zs[0] * fh + ys[1])*fw + xs[0]];\
result += sobelFilter[(z[4] * 223 + y[4]) * 3 + x[4]] * volume[(zs[0] * fh + ys[1])*fw + xs[1]];\
result += sobelFilter[(z[5] * 233 + y[5]) * 3 + x[5]] * volume[(zs[0] * fh + ys[1])*fw + xs[2]];\
result += sobelFilter[(z[6] * 243 + y[6]) * 3 + x[6]] * volume[(zs[0] * fh + ys[2])*fw + xs[0]];\
result += sobelFilter[(z[7] * 253 + y[7]) * 3 + x[7]] * volume[(zs[0] * fh + ys[2])*fw + xs[1]];\
result += sobelFilter[(z[8] * 263 + y[8]) * 3 + x[8]] * volume[(zs[0] * fh + ys[2])*fw + xs[2]];\
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



#define applyKernel(x,y,z) \
result += sobelFilter[(x * 3 + x) * 3 + x] * volume[(zs[0] * fh + ys[0])*fw + xs[0]];\
result += sobelFilter[(x * 3 + x) * 3 + y] * volume[(zs[0] * fh + ys[0])*fw + xs[1]];\
result += sobelFilter[(x * 3 + x) * 3 + z] * volume[(zs[0] * fh + ys[0])*fw + xs[2]];\
result += sobelFilter[(x * 3 + y) * 3 + x] * volume[(zs[0] * fh + ys[1])*fw + xs[0]];\
result += sobelFilter[(x * 3 + y) * 3 + y] * volume[(zs[0] * fh + ys[1])*fw + xs[1]];\
result += sobelFilter[(x * 3 + y) * 3 + z] * volume[(zs[0] * fh + ys[1])*fw + xs[2]];\
result += sobelFilter[(x * 3 + z) * 3 + x] * volume[(zs[0] * fh + ys[2])*fw + xs[0]];\
result += sobelFilter[(x * 3 + z) * 3 + y] * volume[(zs[0] * fh + ys[2])*fw + xs[1]];\
result += sobelFilter[(x * 3 + z) * 3 + z] * volume[(zs[0] * fh + ys[2])*fw + xs[2]];\
	\
result += sobelFilter[(y * 3 + x) * 3 + x] * volume[(zs[1] * fh + ys[0])*fw + xs[0]];\
result += sobelFilter[(y * 3 + x) * 3 + y] * volume[(zs[1] * fh + ys[0])*fw + xs[1]];\
result += sobelFilter[(y * 3 + x) * 3 + z] * volume[(zs[1] * fh + ys[0])*fw + xs[2]];\
result += sobelFilter[(y * 3 + y) * 3 + x] * volume[(zs[1] * fh + ys[1])*fw + xs[0]];\
result += sobelFilter[(y * 3 + y) * 3 + y] * volume[(zs[1] * fh + ys[1])*fw + xs[1]];\
result += sobelFilter[(y * 3 + y) * 3 + z] * volume[(zs[1] * fh + ys[1])*fw + xs[2]];\
result += sobelFilter[(y * 3 + z) * 3 + x] * volume[(zs[1] * fh + ys[2])*fw + xs[0]];\
result += sobelFilter[(y * 3 + z) * 3 + y] * volume[(zs[1] * fh + ys[2])*fw + xs[1]];\
result += sobelFilter[(y * 3 + z) * 3 + z] * volume[(zs[1] * fh + ys[2])*fw + xs[2]];\
	\
result += sobelFilter[(z * 3 + x) * 3 + x] * volume[(zs[2] * fh + ys[0])*fw + xs[0]];\
result += sobelFilter[(z * 3 + x) * 3 + y] * volume[(zs[2] * fh + ys[0])*fw + xs[1]];\
result += sobelFilter[(z * 3 + x) * 3 + z] * volume[(zs[2] * fh + ys[0])*fw + xs[2]];\
result += sobelFilter[(z * 3 + y) * 3 + x] * volume[(zs[2] * fh + ys[1])*fw + xs[0]];\
result += sobelFilter[(z * 3 + y) * 3 + y] * volume[(zs[2] * fh + ys[1])*fw + xs[1]];\
result += sobelFilter[(z * 3 + y) * 3 + z] * volume[(zs[2] * fh + ys[1])*fw + xs[2]];\
result += sobelFilter[(z * 3 + z) * 3 + x] * volume[(zs[2] * fh + ys[2])*fw + xs[0]];\
result += sobelFilter[(z * 3 + z) * 3 + y] * volume[(zs[2] * fh + ys[2])*fw + xs[1]];\
result += sobelFilter[(z * 3 + z) * 3 + z] * volume[(zs[2] * fh + ys[2])*fw + xs[2]];\
	\
	\
result1 += sobelFilter[(x * 3 + x) * 3 + x] * volume[(zs[2] * fh + ys[0])*fw + xs[0]];\
result1 += sobelFilter[(x * 3 + x) * 3 + y] * volume[(zs[2] * fh + ys[0])*fw + xs[1]];\
result1 += sobelFilter[(x * 3 + x) * 3 + z] * volume[(zs[2] * fh + ys[0])*fw + xs[2]];\
result1 += sobelFilter[(x * 3 + y) * 3 + x] * volume[(zs[2] * fh + ys[1])*fw + xs[0]];\
result1 += sobelFilter[(x * 3 + y) * 3 + y] * volume[(zs[2] * fh + ys[1])*fw + xs[1]];\
result1 += sobelFilter[(x * 3 + y) * 3 + z] * volume[(zs[2] * fh + ys[1])*fw + xs[2]];\
result1 += sobelFilter[(x * 3 + z) * 3 + x] * volume[(zs[2] * fh + ys[2])*fw + xs[0]];\
result1 += sobelFilter[(x * 3 + z) * 3 + y] * volume[(zs[2] * fh + ys[2])*fw + xs[1]];\
result1 += sobelFilter[(x * 3 + z) * 3 + z] * volume[(zs[2] * fh + ys[2])*fw + xs[2]];\
	\
result1 += sobelFilter[(y * 3 + x) * 3 + x] * volume[(zs[1] * fh + ys[0])*fw + xs[0]];\
result1 += sobelFilter[(y * 3 + x) * 3 + y] * volume[(zs[1] * fh + ys[0])*fw + xs[1]];\
result1 += sobelFilter[(y * 3 + x) * 3 + z] * volume[(zs[1] * fh + ys[0])*fw + xs[2]];\
result1 += sobelFilter[(y * 3 + y) * 3 + x] * volume[(zs[1] * fh + ys[1])*fw + xs[0]];\
result1 += sobelFilter[(y * 3 + y) * 3 + y] * volume[(zs[1] * fh + ys[1])*fw + xs[1]];\
result1 += sobelFilter[(y * 3 + y) * 3 + z] * volume[(zs[1] * fh + ys[1])*fw + xs[2]];\
result1 += sobelFilter[(y * 3 + z) * 3 + x] * volume[(zs[1] * fh + ys[2])*fw + xs[0]];\
result1 += sobelFilter[(y * 3 + z) * 3 + y] * volume[(zs[1] * fh + ys[2])*fw + xs[1]];\
result1 += sobelFilter[(y * 3 + z) * 3 + z] * volume[(zs[1] * fh + ys[2])*fw + xs[2]];\
	\
result1 += sobelFilter[(z * 3 + x) * 3 + x] * volume[(zs[0] * fh + ys[0])*fw + xs[0]];\
result1 += sobelFilter[(z * 3 + x) * 3 + y] * volume[(zs[0] * fh + ys[0])*fw + xs[1]];\
result1 += sobelFilter[(z * 3 + x) * 3 + z] * volume[(zs[0] * fh + ys[0])*fw + xs[2]];\
result1 += sobelFilter[(z * 3 + y) * 3 + x] * volume[(zs[0] * fh + ys[1])*fw + xs[0]];\
result1 += sobelFilter[(z * 3 + y) * 3 + y] * volume[(zs[0] * fh + ys[1])*fw + xs[1]];\
result1 += sobelFilter[(z * 3 + y) * 3 + z] * volume[(zs[0] * fh + ys[1])*fw + xs[2]];\
result1 += sobelFilter[(z * 3 + z) * 3 + x] * volume[(zs[0] * fh + ys[2])*fw + xs[0]];\
result1 += sobelFilter[(z * 3 + z) * 3 + y] * volume[(zs[0] * fh + ys[2])*fw + xs[1]];\
result1 += sobelFilter[(z * 3 + z) * 3 + z] * volume[(zs[0] * fh + ys[2])*fw + xs[2]];