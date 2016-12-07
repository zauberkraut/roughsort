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
#include <time.h>
#include "sm_20_atomic_functions.h"
#include <iostream>
#include <iomanip>

template<typename T>
class forceMergesort : thrust::binary_function<T, T, bool> {
public:
	__device__ bool operator()(const T& a, const T& b) const {
		return a < b;
	}
};

void devMergesort(int32_t* const a, const int n) {
	thrust::device_ptr<int32_t> devA(a);
	thrust::sort(devA, devA + n, forceMergesort<int32_t>());
	cudaDeviceSynchronize();
}

int devCheckSortedness(int32_t* const a, const int n);

/* Parallel roughsort implementation. */
void devRoughsort(int32_t* const a, const int radius, const int n) {
	devCheckSortedness(a, n);
}

__device__ void devCheckIfSorted(int32_t* a, int n, int id);

__global__ void downInDM(int n, int * b, int * c, int * d)
{

	const int id = blockIdx.x*blockDim.x + threadIdx.x;

	if (id >= n)
		return;

	for (int r = 0; r <= log2((double)n); r++)
	{
		int j = max((double)0, (double)(id - exp2((double)r)));
		if (j <= id && id >= 0 &&
			(j == 0 || c[id] >= b[j - 1])) {
			d[id] = id - j;
			return;
		}
	}
}

__device__ bool sorted;

__global__ void devCheckSortednessCallee(int32_t* const a, const int n, int32_t* b, int32_t* c, int* d)
{
	const int id = blockIdx.x*blockDim.x + threadIdx.x;

	if (id >= n)
		return;

	for (int r = 0; r < log2((double)(n)); r++)
	{
		sorted = true;
		devCheckIfSorted(c, n, id);
		if (sorted == true)
		{
			break;
		}
		else if (id + exp2((double)r) < n)
		{
			int idx = id + exp2((double)r);
			c[id] = min(c[id], c[idx]);
		}
	}

	//max-prefix - 
	for (int r = 0; r < log2((double)(n)); r++)
	{
		sorted = true;
		devCheckIfSorted(b, n, id);
		if (sorted == true)
		{
			break;
		}
		else if (id - exp2((double)r) >= 0)
		{
			int idx = id - exp2((double)r);
			b[id] = max(b[id], b[idx]);
		}
	}
}

__device__ void devCheckIfSorted(int32_t* a, int n, int id)
{

	if (a[id] > a[min(id + 1, n - 1)])
		sorted = false;

}

int devCheckSortedness(int32_t* const a, const int n)
{
	const int nThreadsPerBlock = 1024;
	const int nBlocks = (n + nThreadsPerBlock - 1) / nThreadsPerBlock;

	int32_t * b = (int32_t*)cuMalloc(4 * n);
	int32_t * c = (int32_t*)cuMalloc(4 * n);
	int * d = (int*)cuMalloc(sizeof(int) * n);
	thrust::device_ptr<int> devD(d);

	cudaMemcpy(b, a, 4 * n, cudaMemcpyDeviceToDevice);
	cudaMemcpy(c, a, 4 * n, cudaMemcpyDeviceToDevice);

	clock_t start = clock();
	devCheckSortednessCallee << <nBlocks, nThreadsPerBlock >> >(a, n, b, c, d);
	downInDM << <nBlocks, nThreadsPerBlock >> >(n, b, c, d);

	int k_host = *thrust::max_element(devD, devD + n);
	int ms = (int)(((int64_t)(clock() - start)) * 1000 / CLOCKS_PER_SEC);
	printf("%d\n", ms);

	cuFree((void*)b);
	cuFree((void*)c);
	cuFree((void*)d);

	return k_host;
}
