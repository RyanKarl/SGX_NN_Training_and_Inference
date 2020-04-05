//Enclave.h
//Jonathan S. Takeshita, Ryan Karl, Mark Horeni

#ifndef ENCLAVE_FUNCTIONS_H
#define ENCLAVE_FUNCTIONS_H

#include <assert.h>

#include "Enclave_Defines.h"

//Filler function
inline void rand_bits(int * r, int n){
  assert(r);
  for(int i = 0; i < n; i++){
    r[i] = 0;
    for(unsigned int j = 0; j < sizeof(int)*CHAR_BIT; j++){
      r[i] |= (int)((1 & rand()) << j);
    }
  }
  return;
}

//a, b, c are flattened 2d arrays
//Can consider moving validation outside
int frievald(float * a, float * b, float * c, 
  int a_idx[MAT_DIM], int b_idx[MAT_DIM], int c_idx[MAT_DIM]);

//Return 1 if verification fails, 0 if successful
int verify_frievald(float * data, int a_idx[MAT_DIM], int b_idx[MAT_DIM], int c_idx[MAT_DIM]);

//Return 1 if activation fails, 0 if successful
//Assume data_out is already properly initialized
int activate(float * data_in, int matrix_n[MAT_DIM], 
  float * data_out, int * matrix_n_out);

//Trust that data_in and data_out have the correct size
//Buffers must be allocated outside the enclave!
int verify_and_activate(float * data_in, int a_idx[MAT_DIM], int b_idx[MAT_DIM], int c_idx[MAT_DIM], float ** data_out, int matrix_n_out[MAT_DIM]);

#endif
