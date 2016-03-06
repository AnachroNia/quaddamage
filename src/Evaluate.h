/*!
* \file A simple header to provide all the needed cuda kernels to evaluate an Expression
*/

#ifndef _SURFACE3D_EVALUATE_H_
#define _SURFACE3D_EVALUATE_H_
#include "cuda_runtime.h"

__global__ void Evaluate(float * output, const int * input, const int * types, const float * numbers, int size, int numbers_size, float gridCellWidth, int _x, int _y, int _z, float startX, float startY, float startZ);

#endif