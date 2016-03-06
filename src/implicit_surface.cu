#include "implicit_surface.h"
#include "Expression.h"
#include "Evaluate.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>

#include <thrust/scan.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		system("pause");
		if (abort) exit(code);
	}
}

inline __device__ unsigned int bitCount(unsigned int value) {
	unsigned int count = 0;
	while (value > 0) {           // until all bits are zero
		if ((value & 1) == 1)     // check lower bit
			count++;
		value >>= 1;              // shift bits, removing lower bit
	}
	return count;
}

__global__ void marchingCubes(const float * vertexValues, const int * eTable, const int * tTable, int * vertexCount, int _x, int _y, int _z){
	extern __shared__ int shared_buffer[];
	int * edgeTable = (int *)shared_buffer;
	int * triangleTable = (int *)&edgeTable[256];

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x >= _x) return;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (y >= _y) return;
	int z = blockIdx.z * blockDim.z + threadIdx.z;
	if (z >= _z) return;

	int index = z * (_x * _y) + y*(_x)+x;

	// Shared Memory initialization
	if (index < 256){
		edgeTable[index] = eTable[index];
		for (int i = 0; i < 16; i++){
			triangleTable[index * 16 + i] = tTable[index * 16 + i];
		}
	}
	__syncthreads();
	// End of Shared memory initialization 

	float vertexData[8]; // Load the value of the function in the 8 Vertexes of the Cube
	vertexData[0] = vertexValues[z * (_x * _y) + (y    )*(_x) + (x)];
	vertexData[1] = vertexValues[z * (_x * _y) + (y    )*(_x) + (x + 1)];
	vertexData[2] = vertexValues[z * (_x * _y) + (y + 1)*(_x) + (x + 1)];
	vertexData[3] = vertexValues[z * (_x * _y) + (y + 1)*(_x) + (x)];
	vertexData[4] = vertexValues[(z + 1) * (_x * _y) + (y    )*(_x) + (x)];
	vertexData[5] = vertexValues[(z + 1) * (_x * _y) + (y    )*(_x) + (x + 1)];
	vertexData[6] = vertexValues[(z + 1) * (_x * _y) + (y + 1)*(_x) + (x + 1)];
	vertexData[7] = vertexValues[(z + 1) * (_x * _y) + (y + 1)*(_x) + (x)];

	int cubeIndex = 0;
	int mask = 1;
	for (int i = 0; i < 8; i++){
		if (vertexData[i] < 0) cubeIndex |= mask;
		mask *= 2;
	}

	vertexCount[index] = bitCount(edgeTable[cubeIndex]); // __popc() is undefined?!
	__syncthreads();

	// We use the prefix sum to compress the vertexes array 
	//thrust::exclusive_scan(vertexCount, vertexCount + 6, vertexCount); // in-place scan
	__syncthreads();


}

void ImplicitSurface::generateMesh(){
	//Cell Count in X,Y,Z
	int gridX = (int)gridSize[0] + 1;
	int gridY = (int)gridSize[1] + 1;
	int gridZ = (int)gridSize[2] + 1;

	// Launch Parameters - Not Guaranteed to be optimal 
	dim3 blocks;
	dim3 threads(8, 8, 8); // 512 Threads per Block
	blocks.x = (int)(gridX / 8);
	if (gridX % 8 > 0) blocks.x++;
	blocks.y = (int)(gridY / 8) + 1;
	if (gridY % 8 > 0) blocks.y++;
	blocks.z = (int)(gridZ / 8) + 1;
	if (gridZ % 8 > 0) blocks.z++;

	int cellCount = gridX * gridY * gridZ;
	
	int output_size = functionsCount * cellCount * sizeof(float); // X x Y x Z

	for (int k = 0; k < functionsCount; k++){
		// Allocate Memory on the _device_ 
		int * d_input;
		int input_size = expressions[k]->output.size()*sizeof(int);
		int * d_types;
		int types_size = expressions[k]->type.size()*sizeof(int);
		float * d_output;

		float * d_numbers;
		int numbers_size = expressions[k]->numbers.size()*sizeof(float);

		cudaMalloc((void **)&d_input, input_size);
		cudaMemcpy(d_input, &expressions[k]->output[0], input_size, cudaMemcpyHostToDevice);
		cudaMalloc((void **)&d_types, types_size);
		cudaMemcpy(d_types, &expressions[k]->type[0], types_size, cudaMemcpyHostToDevice);
		cudaMalloc((void **)&d_output, output_size);

		if (expressions[k]->numbers.size() > 0){
			//Allocating 0 bytes causes crash 
			cudaMalloc((void **)&d_numbers, numbers_size);
			cudaMemcpy(d_numbers, &expressions[k]->numbers[0], numbers_size, cudaMemcpyHostToDevice);
		}
		else d_numbers = nullptr;

		//Shared memory alignment 
		//http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#vector-types__alignment-requirements-in-device-code
		int shared_memory_size = expressions[k]->type.size() * sizeof(int) +
			expressions[k]->output.size() * sizeof(int) +
			expressions[k]->numbers.size() * sizeof(float);

		Evaluate << <blocks, threads, shared_memory_size >> >(d_output, d_input, d_types, d_numbers, expressions[k]->type.size(), expressions[k]->numbers.size(), cellSize, gridX, gridY, gridZ,gridStart.x,gridStart.y,gridStart.z);
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());

		float *h_output = (float *)malloc(output_size);
		cudaMemcpy(h_output, d_output, output_size, cudaMemcpyDeviceToHost);
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
	}
}

void ImplicitSurface::beginRender(){ 
	return;
}