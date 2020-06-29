// g++ test_softmax.cpp -Wall -Werror -pedantic -std=c++14

#include <iostream>

#include "Enclave/matrix.h"

using namespace std;

#define ARG_SIZE 3


void print_floatmat(const FP_TYPE * fp, int width, int height){
	for(int j = 0; j < height; j++){
		for(int i = 0; i < width; i++){
			cout << fp[(j*width)+i] << ' ';
		}
		cout << endl;
	}
	cout << endl;
}


int main(int argc, char ** argv){
	//float * arg = (float *) malloc(sizeof(float) * ARG_SIZE);
	/*
	for(int i = 0; i < ARG_SIZE; i++){
		arg[i] = 1.0f + (1.0f*i);
	}
	*/
	FP_TYPE arg[ARG_SIZE*2] = {1.0, 2.0, 3.0, 
	                           4.0, 5.0, 6.0};
	
	
	softmax(arg, ARG_SIZE, 2);
	FP_TYPE * deriv = (FP_TYPE *) malloc(3*3*sizeof(FP_TYPE));
	softmax_derivative(arg, 3, deriv);
	//float * result = softmax_derivative(arg, ARG_SIZE);
	cout << "Derivative:" << endl;
	print_floatmat(deriv, 3, 3);
	assert(is_symmetric(deriv, 3));
	/*
	cout << "Result:" << endl;
	print_floatmat(result, ARG_SIZE, ARG_SIZE);
	*/

	//free(arg);
	//free(result);

	return 0;
}
