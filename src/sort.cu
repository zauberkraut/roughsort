/* sort.cu: CUDA kernels for parallel sorts on an Nvidia GPU. */

#include "thrust/device_ptr.h"
#include "thrust/sort.h"
#include <thrust/extrema.h>
#include "roughsort.h"
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include "math_functions.h"
#include "sm_20_atomic_functions.h"
#include <iostream>


using namespace std;

#define CHECK(r) cuCheck(r, __FILE__, __LINE__) //TODO: get util.h created for windows branch
inline void cuCheck(cudaError_t r, const char* fname, const size_t lnum);
__device__ void devCheckIfSorted(int32_t* a, int n, int local_id, bool * sorted);



void devMergesort(int32_t* const a, const int n) {
  thrust::device_ptr<int32_t> devA(a);
  thrust::stable_sort(devA, devA + n);
}

void devQuicksort(int32_t* const a, const int n) {
  thrust::device_ptr<int32_t> devA(a);
  thrust::sort(devA, devA + n);
}

void devRoughsort(int32_t* const a, const int n) 
{
  thrust::device_ptr<int32_t> devA(a);
  // TODO
}


__global__ void devCheckSortednessCallee(int32_t* const a, const int n, int * k, int * b, int * c, int * d, int * r, bool * sorted)
{


	unsigned long long threadXBits = (unsigned long long)threadIdx.x << (0);
	unsigned long long gridXBits = (unsigned long long)(blockIdx.x) << (10);
	unsigned long long gridYBits = (unsigned long long)(blockIdx.y) << (31 + 10); //arch specific, need to pass the max values
	unsigned long long thread_id = gridXBits | gridYBits | threadXBits;

	int local_id = -1;
	if (thread_id < n)
		local_id = (int)thread_id;
	else
		return;

	b[local_id] = a[local_id];
	c[local_id] = a[local_id];
	d[local_id] = 0;
	//max-prefix - 
	for (int r = 0; r <= log2((float)(n)); r++)
	{
		*sorted = true;
		devCheckIfSorted(b, n, local_id, sorted);
		if (*sorted == true)
		{
			break;
		}
		else if (local_id - exp2((float)r) >= 0)
		{
			int idx = local_id - exp2((float)r);
			b[local_id] = max(b[local_id], b[idx]);
		}
	}
	//min-prefix
	for (int r = 0; r <= log2((float)(n)); r++)
	{
		*sorted = true;
		devCheckIfSorted(c, n, local_id, sorted);
		if (*sorted == true)
		{
			break;
		}
		else if (local_id + exp2((float)r) < n)
		{
			int idx = local_id + exp2((float)r);
			c[local_id] = min(c[local_id], c[idx]);
		}
	}
	
	__syncthreads();

	int i = local_id;
	for (int j = n - 1; j >= 0; j--) {
		if (j <= i && i >= 0 && c[i] <= b[j] &&
			(j == 0 || c[i] >= b[j - 1])) {
			d[i] = i - j;
		}
	}

	//use thrust maxelement to find max of d, which is k
	
	*k = *(thrust::max_element(thrust::device, &d[0], &d[n]));
	

}

__device__ void devCheckIfSorted(int32_t* a, int n, int local_id, bool * sorted)
{

	if (a[local_id] > a[max(local_id + 1, n - 1)])
		*sorted = false;

}

void devCheckSortedness(int32_t* const a, const int n)
{
	
	cudaSetDevice(0);
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);


	unsigned long long max = n;
	unsigned long long threadblockX = max / deviceProp.maxThreadsPerBlock > 1 ? deviceProp.maxThreadsPerBlock : max;
	threadblockX = threadblockX == 0 ? 1 : threadblockX;
	std::cout << "Thread block X: " << threadblockX << std::endl;
	std::cout << "Max block X: " << deviceProp.maxThreadsPerBlock << std::endl;

	unsigned long long threadblockY = 1;
	std::cout << "Thread block Y: " << threadblockY << std::endl;

	unsigned long long threadblockZ = 1;
	std::cout << "Thread block Z: " << threadblockZ << std::endl;

	//calculates required grid X dimension based on the dimension available on device
	unsigned long long gridX = max / (deviceProp.maxGridSize[0]) / threadblockX / threadblockY / threadblockZ > 1 ? deviceProp.maxGridSize[0] : max / threadblockX / threadblockY / threadblockZ + (max % (threadblockX * threadblockY * threadblockZ) > 0 ? 1 : 0);
	gridX = gridX == 0 ? 1 : gridX;
	std::cout << "Grid X: " << gridX << std::endl;
	std::cout << "Max Grid X: " << deviceProp.maxGridSize[0] << std::endl;

	//calculates required grid Y dimension based on the dimension available on device
	unsigned long long gridY = max / threadblockX / threadblockY / threadblockZ / gridX / deviceProp.maxGridSize[1] > 1 ? deviceProp.maxGridSize[1] : max / threadblockX / threadblockY / threadblockZ / gridX + (max % (threadblockX * threadblockY * threadblockZ * gridX) > 0 ? 1 : 0);
	gridY = gridY == 0 ? 1 : gridY;
	std::cout << "Grid Y: " << gridY << std::endl;
	std::cout << "Max Grid Y: " << deviceProp.maxGridSize[1] << std::endl;

	dim3 dimBlock(threadblockX, threadblockY, threadblockZ);
	dim3 dimGrid(gridX, gridY, 1);

	int * b = (int*)cuMalloc(sizeof(int) * n);
	int * c = (int*)cuMalloc(sizeof(int) * n);
	int * d = (int*)cuMalloc(sizeof(int) * n);
	int * r = (int*)cuMalloc(sizeof(int));
	int * k = (int*)cuMalloc(sizeof(int));
	bool * sorted = (bool*)cuMalloc(sizeof(bool));

	int r_host = 0;
	int sorted_host = false;

	cudaMemcpy(r, &r_host, 1, cudaMemcpyHostToDevice);
	cudaMemcpy(sorted, &sorted_host, 1, cudaMemcpyHostToDevice);

	devCheckSortednessCallee << <dimGrid, dimBlock >> >(a, n, k, b, c, d, r, sorted);


	cudaThreadSynchronize();
	cudaDeviceSynchronize();

	int k_host;
	int * b_host = (int*)malloc(sizeof(int)* n);
	int * c_host = (int*)malloc(sizeof(int)* n);
	int * d_host = (int*)malloc(sizeof(int)* n);
	cudaMemcpy(&k_host, k, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(b_host, b, sizeof(int) * n, cudaMemcpyDeviceToHost);
	cudaMemcpy(c_host, c, sizeof(int) * n, cudaMemcpyDeviceToHost);
	cudaMemcpy(d_host, d, sizeof(int) * n, cudaMemcpyDeviceToHost);
	cudaMemcpy(&r_host, r, sizeof(int), cudaMemcpyDeviceToHost);


	CHECK(cudaGetLastError());
	std::cout << "K value: " << k_host << std::endl;
	std::cout << "R value: " << r_host << std::endl;
	for (int i = 0; i < n; i++)
	{
		cout << b_host[i] << "\t" << c_host[i] << "\t" << d_host[i] << endl;
	}
	
	cuFree(b);
	CHECK(cudaGetLastError());
	cuFree(c);
	CHECK(cudaGetLastError());
	cuFree(d);
	CHECK(cudaGetLastError());
	cuFree(r);
	CHECK(cudaGetLastError());
	cuFree(k);
	CHECK(cudaGetLastError());


}


