// g++ test_softmax.cpp -Wall -Werror -pedantic -std=c++14

#include <iostream>

#include "Enclave/matrix.h"

using namespace std;

#define ARG_SIZE 3

void print_floatmat(const float * fp, int height, int width){
	for(int j = 0; j < height; j++){
		for(int i = 0; i < width; i++){
			cout << fp[(j*width)+i] << ' ';
		}
		cout << endl;
	}
	cout << endl;
}


int main(int argc, char ** argv){
	float * arg = (float *) malloc(sizeof(float) * ARG_SIZE);
	for(int i = 0; i < ARG_SIZE; i++){
		arg[i] = 1.0f + (1.0f*i);
	}
	softmax(arg, ARG_SIZE);
	float * result = softmax_derivative(arg, ARG_SIZE);
	cout << "Original:" << endl;
	print_floatmat(arg, 1, ARG_SIZE);
	cout << "Result:" << endl;
	print_floatmat(result, ARG_SIZE, ARG_SIZE);

	free(arg);
	free(result);

	return 0;
}