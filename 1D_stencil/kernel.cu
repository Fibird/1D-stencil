#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <algorithm>

using namespace std;

#define N          1024
#define RADIUS     3
#define BLOCK_SIZE 16

__global__ void stencil_1D(int *in, int *out)
{
	__shared__ int temp[BLOCK_SIZE + 2 * RADIUS];
	int gindex = threadIdx.x + blockDim.x * blockIdx.x;
	int lindex = threadIdx.x + RADIUS;
	// Reads input elements into shared memory
	temp[lindex] = in[gindex];
	if (threadIdx.x < RADIUS)
	{
		temp[lindex - RADIUS] = in[gindex - RADIUS];
		temp[lindex + BLOCK_SIZE] = in[gindex + BLOCK_SIZE];
	}
	// Synchronizes(ensure all the data is available)
	__syncthreads();
	// Applies the stencil
	int result = 0;
	for (int offset = -RADIUS; offset <= RADIUS; offset++)
		result += temp[lindex + offset];
	// Stores the result
	out[gindex] = result;
}


int main()
{
	int *in, *out;
	int *dev_in, *dev_out;

	// Allocs space for host copies and setup values
	in = (int*)malloc((N + 2 * RADIUS) * sizeof(int));
	fill_n(in, (N + 2 * RADIUS), 1);
	//fill_n(in, RADIUS, 0);
	//fill_n(in + N + RADIUS, RADIUS, 0);
	out = (int*)malloc((N + 2 * RADIUS) * sizeof(int));
	fill_n(out, (N + 2 * RADIUS), 1);
	//fill_n(out, RADIUS, 0);
	//fill_n(out + N + RADIUS, RADIUS, 0);

	// Alloc space for device copies
	cudaMalloc((void**)&dev_in, (N + 2 * RADIUS) * sizeof(int));
	cudaMalloc((void**)&dev_out, (N + 2 * RADIUS) * sizeof(int));

	// Copies to device
	cudaMemcpy(dev_in, in, (N + 2 * RADIUS) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_out, out, (N + 2 * RADIUS) * sizeof(int), cudaMemcpyHostToDevice);

	// Launches stencil 1D kernel on GPU
	stencil_1D<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(dev_in + RADIUS, dev_out + RADIUS);

	// Copies back to host
	cudaMemcpy(out, dev_out, (N + 2 * RADIUS) * sizeof(int), cudaMemcpyDeviceToHost);

	for (int i = RADIUS; i < N; i++)
		cout << out[i] << " ";
	cout << endl;

	// cleanup
	free(in);
	free(out);
	cudaFree(dev_in);
	cudaFree(dev_out);
}

