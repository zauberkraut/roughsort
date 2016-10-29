/* sort.cu: CUDA kernels for parallel sorts on an Nvidia GPU. */

#include "thrust/device_ptr.h"
#include "thrust/sort.h"
#include "roughsort.h"
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include "math_functions.h"
#include "sm_20_atomic_functions.h"
#include <iostream>

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



__global__ void devCheckSortednessCallee(int32_t* const a, const int n, int * k, bool * sorted)
{

	
	unsigned long long threadXBits = (unsigned long long)threadIdx.x << (0);
	unsigned long long gridXBits = (unsigned long long)(blockIdx.x) << (10);
	unsigned long long gridYBits = (unsigned long long)(blockIdx.y) << (31 + 10); //arch specific, need to pass the max values
	unsigned long long thread_id = gridXBits | gridYBits | threadXBits;
	
	int local_id = -1;
	if (thread_id < n)
		local_id = (int)thread_id;
	
	int i = 1;
	for (; i < n && local_id < n && local_id >= 0 && *sorted == false; i++) //naive k check
	{
		*sorted = true;
		if (local_id + i < n && a[local_id] > a[local_id + i])
			*sorted = false;
		

		if (local_id - i >= 0 && a[local_id] < a[local_id - i])
			*sorted = false;
		
		if (*sorted == true)
			*k = i;
	}


	if (*sorted == false)
		*k = -n;
	
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

	int * k_dev = (int*)cuMalloc(sizeof(int));
	bool * sorted_dev = (bool*)cuMalloc(sizeof(bool));
	bool sorted_host = false;
	int k_val = -1;
	CHECK(cudaMemcpy((void*)sorted_dev, (void*)&sorted_host, sizeof(bool), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy((void*)k_dev, (void*)&k_val, sizeof(int), cudaMemcpyHostToDevice));


	devCheckSortednessCallee << <dimGrid, dimBlock >> >(a, n, k_dev, sorted_dev);


	cudaThreadSynchronize();
	cudaDeviceSynchronize();

	int k_host;
	cudaMemcpy(&k_host, k_dev, sizeof(int), cudaMemcpyDeviceToHost);

	CHECK(cudaGetLastError());
	std::cout << "K value: " << k_host << std::endl;




}


