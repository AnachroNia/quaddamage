#include "Evaluate.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>

#include <math.h>
#include <stdio.h>

/*! 
* \brief Evaluates a serialized math expression using Dijkstra's Shunting Yard Algorithm at 
* points of a grid
* \param[out] out An output array - it will contain the values of the function
* at 
* \param[in] input An array of operation numbers serialized by the Expression class
*                  If the token type is a number - the input represents the index of this number
*                  in the numbers array. 
*                  For all other token types - input gives us an index in the operations table
*                  for the specified token type - Check Expression.h Expression::Expression for refference
* \param[in] types An array of token types serialized by the Expression class. 
*                  Check Expression.h TokenTypes enum for refference
* \note Both input & types are used to encode a single operation 
* \param[in] numbers An array of all the constants known before evaluation of the expression
* \param[in] size The number of tokens
* \param[in] numbers_size The number of constants
* \param[in] gridCellWidth The width of the grid - The smaller the value, the more detail you get
* \param[in] _x, _y, _z - The number of Cells in X, Y and Z dimensions
* \param[in] startX, startY, startZ A point where to position the first Cell of the grid.
*/
__global__ void Evaluate(float * out, const int * input, const int * types, const float * numbers, int size, int numbers_size, float gridCellWidth, int _x, int _y, int _z, float startX, float startY, float startZ) {
	// Move the input to the shared memory for faster access
	extern __shared__ int shared_buffer[];
	int * s_types = (int *)shared_buffer;		
	int * s_input = (int *)&s_types[size];
	float * s_numbers = (float *)&s_input[size];
	
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;

	int index = z * (_x * _y) + y*(_x) + x;

	int shared_index = threadIdx.z * (blockDim.x* blockDim.y) + threadIdx.y*blockDim.x+threadIdx.x;
	if (shared_index < size){
		s_types[shared_index] = types[shared_index];
		s_input[shared_index] = input[shared_index];
	}

	if (shared_index < numbers_size){
		s_numbers[shared_index] = numbers[shared_index];
	}
	__syncthreads(); //Shared memory initialized

	if (x >= _x || y >= _y || z >= _z) return;

#define CU_ARRAY_SIZE 24 //On Kepler you have only 64 registers per thread - gotta be sparing 
	// Output Queue for Shunting Yard Algorithm
	float output[CU_ARRAY_SIZE];
	int output_index = 0;
	// Operators Stack for Shunting Yard Algorithm
	int operators[CU_ARRAY_SIZE];
	int operators_index = 0;

	for (int i = 0; i < size; i++){
		int operation = s_types[i]; //Use local memory/registers instead of shared memory 
		if (operation == 1){ // Number 
			output[output_index] = s_numbers[s_input[i]];
			output_index++;
		}
		else if (operation == 2){ // Variable
			float var = s_input[i];
			if (var == 0){
				var = startX + gridCellWidth * x;
			}
			else if (var == 1){
				var = startY + gridCellWidth * y;
			}
			else if (var == 2){
				var = startZ + gridCellWidth * z; 
			}
			else if (var == 3){
				//TODO - time implementation 
				var = 0;
			}
			output[output_index] = var;
			output_index++;
		}
		else if (operation == 4){ //Function 
			operators[operators_index] = i; //Store the index so we can look up in the types array
			operators_index++;
		}
		else if (operation == 3){ //Operator
			int op = s_input[i];
			int it = 0;
			if(operators_index > 0 ) it = operators[operators_index - 1];
			while (operators_index > 0 && s_types[it] == 3 && op <= s_input[it]){
					if (s_input[it] == 0){ //+
						output[output_index - 2] += output[output_index - 1];
					}
					else if (s_input[it] == 1){ //- 
						output[output_index - 2] -= output[output_index - 1];
					}
					else if (s_input[it] == 2){ // / 
						output[output_index - 2] /= output[output_index - 1];
					}
					else if (s_input[it] == 3){ //*
						output[output_index - 2] *= output[output_index - 1];
					}
					else if (s_input[it] == 4){ //%
						output[output_index - 2] = fmod(output[output_index - 2], output[output_index - 1]);
					}
					else if (s_input[it] == 5){ //^
						output[output_index - 2] = pow(output[output_index - 2], output[output_index - 1]);
					}
					// Remove the operator and the last number
					output_index--;
					operators_index--;
					it = operators[operators_index - 1];
			} //End While
			// Push the new operator on the stack 
			operators[operators_index] = i;
			operators_index++;
		}
		else if (operation == 5){ // Brackets
			int bracket = s_input[i];
			if (bracket == 0){ //Openning 
				operators[operators_index] = i;
				operators_index++;
			}
			else { //Closing Bracket
				while ((s_types[operators[operators_index - 1]] != 5)){
					int op = s_input[operators[operators_index - 1]];
					if (op == 0){ //+
						output[output_index - 2] += output[output_index - 1];
					}
					else if (op == 1){ //- 
						output[output_index - 2] -= output[output_index - 1];
					}
					else if (op == 2){ // / 
						output[output_index - 2] /= output[output_index - 1];
					}
					else if (op == 3){ //*
						output[output_index - 2] *= output[output_index - 1];
					}
					else if (op == 4){ //%
						output[output_index - 2] = fmod(output[output_index - 2], output[output_index - 1]);
					}
					else if (op == 5){ //^
						output[output_index - 2] = pow(output[output_index - 2], output[output_index - 1]);
					}
					// Remove the operator and the last number
					output_index--;
					operators_index--;
				}
				operators_index--; //Remove the opening bracket 
				if (operators_index > 0 && s_types[operators[operators_index - 1]] == 4){ //Function before the brackets?
					int func = s_input[operators[operators_index - 1]];
					if (func == 1){
						output[output_index - 1] = sin(output[output_index - 1]);
					}
					else if (func == 2){
						output[output_index - 1] = cos(output[output_index - 1]);
					}
					else if (func == 3){
						output[output_index - 1] = tan(output[output_index - 1]);
					}
					else if (func == 4){
						output[output_index - 1] = 1 / tan(output[output_index - 1]);
					}
					else if (func == 5){
						output[output_index - 1] = asin(output[output_index - 1]);
					}
					else if (func == 6){
						output[output_index - 1] = acos(output[output_index - 1]);
					}
					else if (func == 7){
						output[output_index - 1] = atan(output[output_index - 1]);
					}
					else if (func == 8){
						output[output_index - 1] = atan(1 / (output[output_index - 1]));
					}
					else if (func == 9){
						output[output_index - 1] = sqrt(output[output_index - 1]);
					}
					operators_index--;
				} //End function if
			} //End closing brackets 
		} //End brackets if 
	} // End - All Tokens read

	while (operators_index > 0){ //Use any operators left on the stack
		int op = s_input[operators[operators_index - 1]];
		if (op == 0){ //+
			output[output_index - 2] += output[output_index - 1];
		}
		else if (op == 1){ //- 
			output[output_index - 2] -= output[output_index - 1];
		}
		else if (op == 2){ // / 
			output[output_index - 2] /= output[output_index - 1];
		}
		else if (op == 3){ //*
			output[output_index - 2] *= output[output_index - 1];
		}
		else if (op == 4){ //%
			output[output_index - 2] = fmod(output[output_index - 2], output[output_index - 1]);
		}
		else if (op == 5){ //^
			output[output_index - 2] = pow(output[output_index - 2], output[output_index - 1]);
		}
		// Remove the operator and the last number
		output_index--;
		operators_index--;
	}
	// Calculations finished - The end result is in output[0] 
	out[index] = output[0];
}
