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


using namespace std;

#define CHECK(r) cuCheck(r, __FILE__, __LINE__) //TODO: get util.h created for windows branch
inline void cuCheck(cudaError_t r, const char* fname, const size_t lnum);
__device__ void devCheckIfSorted(int32_t* a, int n, int local_id, bool * sorted);

void devRadixsort(int32_t* const a, const int n) {
	thrust::device_ptr<int32_t> devA(a);
	thrust::sort(devA, devA + n);
	cudaDeviceSynchronize();
}

void devMergesort(int32_t* const a, const int n) {
  thrust::device_ptr<int32_t> devA(a);
  thrust::stable_sort(devA, devA + n);
}

void devQuicksort(int32_t* const a, const int n) {
  thrust::device_ptr<int32_t> devA(a);
  thrust::sort(devA, devA + n);
}

static __global__ void kernHalve(int32_t* const a, const int radius,
	const int n) {
	const int i = min(n, radius*(blockIdx.x*blockDim.x + threadIdx.x));
	const int j = min(n, i + radius);
	const int k = min(n, i + (radius >> 1));
	const int l = min(n, k + radius);

	thrust::sort(thrust::seq, a + i, a + j);
	thrust::sort(thrust::seq, a + k, a + l);
	thrust::sort(thrust::seq, a + i, a + j);
}

/* Parallel roughsort implementation.
ASSUMES k > 1 */
void devRoughsort(int32_t* const a, const int radius, const int n) {
	if (!radius || n < 2) {
		return;
	}

	cudaError_t r;
	int k = radius, p = 0;
	do {
		const int nSegments = (n + k - 1) / k;
		const int nBlocks = (nSegments + 511) / 512;
		kernHalve << <nBlocks, 512 >> >(a, k, n);
		r = cudaGetLastError();

		k = radius / (2 << p++);
	} while (k > 1 && r == cudaSuccess);

	if (r != cudaSuccess) {
		printf("KERNEL ERROR: %s\n", cudaGetErrorString(r));
	}
	cudaDeviceSynchronize();
}


__global__ void downInDM(int n, int * b, int * c, int * d)
{

	const int thread_id = blockIdx.x*blockDim.x + threadIdx.x;

	int local_id = 0;
	if (thread_id < n)
		local_id = thread_id;
	else
		return;

	__syncthreads();
	__threadfence();
	int i = local_id;

	for (int r = 0; r <= log2((double)n); r++)
	{
		int j = max((double)0, (double)(i - exp2((double)r)));
		//if (j <= i && i >= 0 && c[i] <= b[j] && -- note the removed criteria from seq
		if (j <= i && i >= 0 &&
			(j == 0 || c[i] >= b[j - 1])) {
			d[i] = i - j;
			return;

		}
		__threadfence();
		__syncthreads();
	}
}

__global__ void devCheckSortednessCallee(int32_t* const a, const int n, int * k, int * b, int * c, int * d, int * r, int * l, int * m, int * o, int * p, bool * sorted, int tpbBits, int g0Bits, int g1Bits)
{
	
	const int thread_id = blockIdx.x*blockDim.x + threadIdx.x;

	int local_id = 0;
	if (thread_id < n)
		local_id = thread_id;
	else
		return;

	atomicAdd(r, thread_id);
	

	
	b[local_id] = a[local_id];
	c[local_id] = a[local_id];
	d[local_id] = 0;
	l[local_id] = 0;
	m[local_id] = 0;
	
	//max-prefix - 
	for (int r = 0; r < log2((double)(n)); r++)
	{
		*sorted = true;
		__threadfence();
		devCheckIfSorted(b, n, local_id, sorted);
		__threadfence();
		if (*sorted == true)
		{
			break;
		}
		else if (local_id - exp2((double)r) >= 0)
		{
			int idx = local_id - exp2((double)r);
			b[local_id] = max(b[local_id], b[idx]);
		}
	}
	//min-prefix
	for (int r = 0; r < log2((double)(n)); r++)
	{

		*sorted = true;
		__threadfence();
		devCheckIfSorted(c, n, local_id, sorted);
		__threadfence();
		if (*sorted == true)
		{
			break;
		}
		else if (local_id + exp2((double)r) < n)
		{
			int idx = local_id + exp2((double)r);
			c[local_id] = min(c[local_id], c[idx]);
		}
	}
	
	__syncthreads();

	/*
	if (c[local_id] < b[local_id])
		l[local_id] = 1;

	if (local_id == 0 || c[local_id] > b[max(0, local_id - 1)])
		m[local_id] = 1;

	if (c[local_id] == b[local_id])
	{
		o[local_id] = 1;
	}

	if (local_id - 1 >= 0 && c[local_id] == b[max(0, local_id - 1)])
	{
		p[local_id] = 1;
	}*/




	/*
	int j = i;
	for (int r = log2((double)n); r >= 0; r--)
	{
			
			
			if (j <= i && i >= 0 && c[i] <= b[j] &&
			//if (j <= i && i >= 0 &&
				(j == 0 || c[i] >= b[j - 1])) {
				d[i] = i - j;
				return;
			}
			else {
				j = c[i] <= b[j] ? max((double)0, (double)(j - exp2((double)r))) : min((double)n - 1, j + exp2((double)r));
			}
	}*/



}

__device__ void devCheckIfSorted(int32_t* a, int n, int local_id, bool * sorted)
{

	if (a[local_id] > a[min(local_id + 1, n - 1)])
		*sorted = false;

}

void devCheckSortedness(int32_t* const a, const int n)
{
	
	cudaSetDevice(0);
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
	cout << "using " << deviceProp.multiProcessorCount << " multiprocessors" << endl;
	cout << "max threads per processor: " << deviceProp.maxThreadsPerMultiProcessor << endl;

	unsigned long long max = n;
	unsigned long long threadblockX = max / deviceProp.maxThreadsPerBlock > 1 ? deviceProp.maxThreadsPerBlock : max;
	threadblockX = threadblockX == 0 ? 1 : threadblockX;
	std::cout << "Thread block X: " << threadblockX << std::endl;
	//std::cout << "Max block X: " << deviceProp.maxThreadsPerBlock << std::endl;

	unsigned long long threadblockY = 1;
	//std::cout << "Thread block Y: " << threadblockY << std::endl;

	unsigned long long threadblockZ = 1;
	//std::cout << "Thread block Z: " << threadblockZ << std::endl;

	//calculates required grid X dimension based on the dimension available on device
	unsigned long long gridX = max / (deviceProp.maxGridSize[0]) / threadblockX / threadblockY / threadblockZ > 1 ? deviceProp.maxGridSize[0] : max / threadblockX / threadblockY / threadblockZ + (max % (threadblockX * threadblockY * threadblockZ) > 0 ? 1 : 0);
	gridX = gridX == 0 ? 1 : gridX;
	std::cout << "Grid X: " << gridX << std::endl;
	//std::cout << "Max Grid X: " << deviceProp.maxGridSize[0] << std::endl;

	//calculates required grid Y dimension based on the dimension available on device
	unsigned long long gridY = max / threadblockX / threadblockY / threadblockZ / gridX / deviceProp.maxGridSize[1] > 1 ? deviceProp.maxGridSize[1] : max / threadblockX / threadblockY / threadblockZ / gridX + (max % (threadblockX * threadblockY * threadblockZ * gridX) > 0 ? 1 : 0);
	gridY = gridY == 0 ? 1 : gridY;
	std::cout << "Grid Y: " << gridY << std::endl;
	//std::cout << "Max Grid Y: " << deviceProp.maxGridSize[1] << std::endl;

	const int nThreadsPerBlock = 1024;
	const int nBlocks = (n + nThreadsPerBlock - 1) / nThreadsPerBlock;


	int * b = (int*)cuMalloc(sizeof(int) * n);
	int * c = (int*)cuMalloc(sizeof(int) * n);
	int * d = (int*)cuMalloc(sizeof(int) * n);
	int * l = (int*)cuMalloc(sizeof(int) * n);
	int * m = (int*)cuMalloc(sizeof(int) * n);
	int * o = (int*)cuMalloc(sizeof(int) * n);
	int * p = (int*)cuMalloc(sizeof(int) * n);
	int * r = (int*)cuMalloc(sizeof(int));
	int * k = (int*)cuMalloc(sizeof(int));
	bool * sorted = (bool*)cuMalloc(sizeof(bool));


	int r_host = 0;
	int sorted_host = false;

	CHECK(cudaMemcpy(r, &r_host, 1, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(sorted, &sorted_host, 1, cudaMemcpyHostToDevice));


	clock_t tStart = clock();

	devCheckSortednessCallee << <nBlocks, nThreadsPerBlock >> >(a, n, k, b, c, d, r, l, m, o, p, sorted, 
		log2(deviceProp.maxThreadsPerBlock), 
		log2(deviceProp.maxGridSize[0]), log2(deviceProp.maxGridSize[1]));
	printf("Err: %s", cudaGetErrorString(cudaGetLastError()));
	cudaThreadSynchronize();
	printf("Err: %s", cudaGetErrorString(cudaGetLastError()));
	cudaDeviceSynchronize();
	printf("Err: %s", cudaGetErrorString(cudaGetLastError()));
	downInDM << <nBlocks, nThreadsPerBlock >> >(n, b, c, d);

	printf("Err: %s", cudaGetErrorString(cudaGetLastError()));
	cudaThreadSynchronize();
	printf("Err: %s", cudaGetErrorString(cudaGetLastError()));
	cudaDeviceSynchronize();
	printf("Err: %s", cudaGetErrorString(cudaGetLastError()));
	clock_t runtime = (double)(clock() - tStart);

	cout << "K check runs in: " << setiosflags(ios::fixed)
		<< setprecision(4)
		<< runtime << " ticks\n";

	int k_host;
	int * b_host = (int*)malloc(sizeof(int)* n);
	int * c_host = (int*)malloc(sizeof(int)* n);
	int * d_host = (int*)malloc(sizeof(int)* n);
	int * l_host = (int*)malloc(sizeof(int)* n);
	int * m_host = (int*)malloc(sizeof(int) * n);
	int * o_host = (int*)malloc(sizeof(int) * n);
	int * p_host = (int*)malloc(sizeof(int) * n);
	int * a_host = (int*)malloc(sizeof(int) * n);
	cudaMemcpy(b_host, b, sizeof(int) * n, cudaMemcpyDeviceToHost);
	cudaMemcpy(c_host, c, sizeof(int) * n, cudaMemcpyDeviceToHost);
	cudaMemcpy(d_host, d, sizeof(int) * n, cudaMemcpyDeviceToHost);
	CHECK(cudaGetLastError());
	cudaMemcpy(&r_host, r, sizeof(int), cudaMemcpyDeviceToHost);
	CHECK(cudaGetLastError());
	cudaMemcpy(l_host, l, sizeof(int) * n, cudaMemcpyDeviceToHost);
	CHECK(cudaGetLastError());
	cudaMemcpy(m_host, m, sizeof(int) * n, cudaMemcpyDeviceToHost);
	CHECK(cudaGetLastError());
	cudaMemcpy(o_host, o, sizeof(int) * n, cudaMemcpyDeviceToHost);
	CHECK(cudaGetLastError());
	cudaMemcpy(p_host, p, sizeof(int) * n, cudaMemcpyDeviceToHost);
	CHECK(cudaGetLastError());
	cudaMemcpy(a_host, a, sizeof(int) * n, cudaMemcpyDeviceToHost);
	CHECK(cudaGetLastError());
	k_host = *(thrust::max_element(thrust::host, &d_host[0], &d_host[n]));


	CHECK(cudaGetLastError());
	std::cout << "K value: " << k_host << std::endl;
	std::cout << "R value: " << r_host << std::endl;
	
	if(n<=256 || n == 4096)
	for (int i = 0; i < n; i++)
	{
		cout << a_host[i] << "\t" << i << b_host[i] << "\t" << c_host[i] << "\t" << d_host[i] << "\t" << l_host[i] << "\t" << o_host[i] << "\t" << m_host[i] << "\t" << p_host[i] << endl;
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
	cuFree(sorted);
	CHECK(cudaGetLastError());
	cuFree(l);
	CHECK(cudaGetLastError());


}


