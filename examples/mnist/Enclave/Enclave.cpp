//enclave_functions.c
//Jonathan S. Takeshita, Ryan Karl, Mark Horeni

//#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>
#include <math.h>

#include "Enclave.h"

int frievald(float * a, float * b, float * c, 
  int a_idx[MAT_DIM], int b_idx[MAT_DIM], int c_idx[MAT_DIM])
{
  //Mult. is defined
  assert(a_idx[1] == b_idx[0]);
  //Output dims are correct
  assert(c_idx[0] == a_idx[0]);
  assert(c_idx[1] == b_idx[1]);
  //Create a random vector r
  size_t num_bytes_randarr = (b_idx[1]/CHAR_BIT) + (b_idx[1]%CHAR_BIT? 1 : 0);
  unsigned char * r = (unsigned char *) calloc(num_bytes_randarr, sizeof(unsigned char));
  if(!r){
    assert(r && "calloc failed");
  }
  rand_bytes(r, num_bytes_randarr);

  //Hope that calloc properly sets bits to 0
  //Calculate br, cr in the same loop
  float * br = (float *) calloc(b_idx[1], sizeof(float));
  float * cr = (float *) calloc(c_idx[1], sizeof(float));
  if(!br){
    assert(br && "malloc failed");
  }
  if(!cr){
    assert(cr && "malloc failed");
  }
  for (int i = 0; i < b_idx[0]; i++){
      for (int j = 0; j < b_idx[1]; j++){
          br[i] += INDEX_FLOATMAT(b, i, j, b_idx[0]) * ((unsigned char)INDEX_BITARR(r, j));
      }
  }

  for (int i = 0; i < c_idx[0]; i++){
      for (int j = 0; j < c_idx[1]; j++){
          cr[i] += INDEX_FLOATMAT(c, i, j, c_idx[0]) * ((unsigned char)INDEX_BITARR(r, j));
      }
  }

  free(r);
  r = NULL;

  float * axbr = (float *) calloc(b_idx[0], sizeof(float));
  if(!axbr){
    assert(axbr && "malloc failed");
  }
  assert(axbr && "Allocating axbr failed!");
  for (int i = 0; i < b_idx[0]; i++){
      for (int j = 0; j < b_idx[1]; j++){
          axbr[i] += INDEX_FLOATMAT(a, i, j, b_idx[0]) * br[j];
      }
  }

  free(br);
  br = NULL;

  for (int i = 0; i < c_idx[1]; i++){
    if (FLOAT_CMP(axbr[i], cr[i])){
        free(axbr);
        free(cr);
        axbr = cr = NULL;
        return 1;
    }
  }

  free(axbr);
  axbr = NULL;
  free(cr);
  cr = NULL;

  return 0;
}

int verify_frievald(float * data, int a_idx[MAT_DIM], int b_idx[MAT_DIM], int c_idx[MAT_DIM]){
  float * matrix_offsets[NUM_MATRICES];
  matrix_offsets[0] = data;
  matrix_offsets[1] = matrix_offsets[0] + (a_idx[0]*a_idx[1]);
  matrix_offsets[2] = matrix_offsets[1] + (b_idx[0]*b_idx[1]);
  for(unsigned int j = 0; j < K_PROBABILITY; j++){
  	if(frievald(matrix_offsets[0], matrix_offsets[1], matrix_offsets[2], a_idx, b_idx, c_idx)){
  		return 1;
  	}
  }
  return 0;
}

int activate(float * data_in, int matrix_n[MAT_DIM], 
  float * data_out, int * matrix_n_out){
  //Use the below if things are done in-place
  //data_outshape must have DATA_DIMENSIONS elements
  for(unsigned int i = 0; i < MAT_DIM; i++){
    matrix_n_out[i] = matrix_n[i];
  }

  //Using tanh as the activation function
  for(int j = 0; j < matrix_n[0]*matrix_n[1]; j++){
    data_out[j] = tanh(data_in[j]);
  }  

  return 0;
}

int verify_and_activate(float * data_in, int a_idx[MAT_DIM], int b_idx[MAT_DIM], int c_idx[MAT_DIM], float ** data_out, int matrix_n_out[MAT_DIM]){
  //Copy data to enclave space
  int mult_ptr_offset = (a_idx[0]*a_idx[1]) + (b_idx[0]*b_idx[1]);
  int total_input_elts = mult_ptr_offset + (c_idx[0]*c_idx[1]);
  float * enclave_data = (float *) malloc(total_input_elts*sizeof(float));
  for(int i = 0; i < total_input_elts; i++){
    enclave_data[i] = data_in[i];
  }
  
  if(verify_frievald(enclave_data, a_idx, b_idx, c_idx)){
    free(enclave_data);
    enclave_data = NULL;
    return 1;
  }
  
  if(!*data_out){
    *data_out = data_in + mult_ptr_offset;
  }
  
  float * activation_buffer_enclave = enclave_data + mult_ptr_offset;
  if(activate(activation_buffer_enclave, c_idx, *data_out, matrix_n_out)){
    free(enclave_data);
    enclave_data = NULL;
    return 1;
  }
  
  free(enclave_data);
  enclave_data = NULL;
  return 0;

}
