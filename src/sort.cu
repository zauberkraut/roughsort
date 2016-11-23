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
#include "Lock.h"

using namespace std;

#define CHECK(r) cuCheck(r, __FILE__, __LINE__) //TODO: get util.h created for windows branch
inline void cuCheck(cudaError_t r, const char* fname, const size_t lnum);


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


__global__ void devCheckSortednessCallee(int32_t* const a, const int n, int * k, int * b, int * c, int * d, int * r, Lock lock)
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
	
	//max-prefix - 
	for (int r = 0; r < log2((float)n); r++)
	{
		bool sorted = true;
		for (int i = 1; i < n; i++)
		{

			if (a[i] < a[i - 1])
				sorted = false;
		}
		if (sorted == true)
		{
			break;
		}
		else if (local_id - exp2((float)r) >= 0)
		{
			int idx = local_id - exp2((float)r);
			b[local_id] = max(a[local_id], a[idx]);
		}
	}
	//min-prefix
	for (int r = 0; r < log2((float)n); r++)
	{
		bool sorted = true;
		for (int i = 1; i < n; i++)
		{

			if (a[i] < a[i - 1])
				sorted = false;
		}
		if (sorted == true)
		{
			break;
		}
		else if (local_id - exp2((float)r) >= 0)
		{
			int idx = local_id - exp2((float)r);
			c[local_id] = min(a[local_id], a[idx]);
		}
	}
	//d
	int i = n;
	while ((local_id <= i) && (i > 0) && c[i] <= b[local_id] && ((local_id == 0) || (c[i] >= b[local_id - 1])))
	{
		lock.lock();
		d[i] = i - local_id;
		lock.unlock();
		i--;
	}

	//use thrust maxelement to find max of d, which is k

	thrust::device_ptr<int32_t> d_start = thrust::device_pointer_cast((int32_t*)d);
	thrust::device_ptr<int32_t> d_end = thrust::device_pointer_cast((int32_t*)(d + n - 1));
	*k = *(thrust::max_element(thrust::device, d_start, d_end));

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
	Lock lock;


	devCheckSortednessCallee << <dimGrid, dimBlock >> >(a, n, k, b, c, d, r, lock);


	cudaThreadSynchronize();
	cudaDeviceSynchronize();

	int k_host;
	cudaMemcpy(&k_host, k, sizeof(int), cudaMemcpyDeviceToHost);

	CHECK(cudaGetLastError());
	std::cout << "K value: " << k_host << std::endl;

	cuFree(b);
	cuFree(c);
	cuFree(d);
	cuFree(r);
	cuFree(k);



}


