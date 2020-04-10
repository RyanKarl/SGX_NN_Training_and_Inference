//enclave_functions.c
//Jonathan S. Takeshita, Ryan Karl, Mark Horeni

//#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>
#include <math.h>

#include "Enclave.h"

//0 is height, 1 is width
int frievald(float * a, float * b, float * c, 
  mat_dim_t a_idx, mat_dim_t b_idx, mat_dim_t c_idx)
{
  //Mult. is defined
  assert(a_idx.width == b_idx.height);
  //Output dims are correct
  assert(c_idx.height == a_idx.height);
  assert(c_idx.width == b_idx.width);
  //Create a random vector r
  size_t num_bytes_randarr = (b_idx.width/CHAR_BIT) + (b_idx.width%CHAR_BIT? 1 : 0);
  unsigned char * r = (unsigned char *) calloc(num_bytes_randarr, sizeof(unsigned char));
  if(!r){
    assert(r && "calloc failed");
  }
  rand_bytes(r, num_bytes_randarr);

  //Hope that calloc properly sets bits to 0
  //Calculate br, cr in the same loop
  float * br = (float *) calloc(b_idx.width, sizeof(float));
  float * cr = (float *) calloc(c_idx.width, sizeof(float));
  if(!br){
    assert(br && "malloc failed");
  }
  if(!cr){
    assert(cr && "malloc failed");
  }
  for (int i = 0; i < b_idx.height; i++){
      for (int j = 0; j < b_idx.width; j++){
          br[i] += INDEX_FLOATMAT(b, i, j, b_idx.height) * ((unsigned char)INDEX_BITARR(r, j));
      }
  }

  for (int i = 0; i < c_idx.height; i++){
      for (int j = 0; j < c_idx.width; j++){
          cr[i] += INDEX_FLOATMAT(c, i, j, c_idx.height) * ((unsigned char)INDEX_BITARR(r, j));
      }
  }

  free(r);
  r = NULL;

  float * axbr = (float *) calloc(b_idx.height, sizeof(float));
  if(!axbr){
    assert(axbr && "malloc failed");
  }
  assert(axbr && "Allocating axbr failed!");
  for (int i = 0; i < b_idx.height; i++){
      for (int j = 0; j < b_idx.width; j++){
          axbr[i] += INDEX_FLOATMAT(a, i, j, b_idx.height) * br[j];
      }
  }

  free(br);
  br = NULL;

  for (int i = 0; i < c_idx.width; i++){
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

int verify_frievald(float * data, mat_dim_t a_idx, mat_dim_t b_idx, mat_dim_t c_idx){
  float * matrix_offsets[NUM_MATRICES];
  matrix_offsets[0] = data;
  matrix_offsets[1] = matrix_offsets[0] + (a_idx.height*a_idx.width);
  matrix_offsets[2] = matrix_offsets[1] + (b_idx.height*b_idx.width);
  for(unsigned int j = 0; j < K_PROBABILITY; j++){
  	if(frievald(matrix_offsets[0], matrix_offsets[1], matrix_offsets[2], a_idx, b_idx, c_idx)){
  		return 1;
  	}
  }
  return 0;
}

int activate(float * data_in, mat_dim_t matrix_n, 
  float * data_out, mat_dim_t matrix_n_out){
  //Use the below if things are done in-place
  //data_outshape must have DATA_DIMENSIONS elements
  /*
  for(unsigned int i = 0; i < MAT_DIM; i++){
    matrix_n_out[i] = matrix_n[i];
  }
  */

  //Using tanh as the activation function
  for(int j = 0; j < matrix_n.height*matrix_n.width; j++){
    data_out[j] = tanh(data_in[j]);
  }  

  return 0;
}

int verify_and_activate(float * data_in, mat_dim_t a_idx, mat_dim_t b_idx, mat_dim_t c_idx,
 float * data_out, mat_dim_t matrix_n_out){
  //Copy data to enclave space
  //Validate data here
  if(a_idx.height < 0 || a_idx.width < 0 ||
     b_idx.height < 0 || b_idx.width < 0 ||
     c_idx.height < 0 || c_idx.width < 0 ||
     matrix_n_out.height < 0 || matrix_n_out.width < 0){
    return 1;
  }
  int mult_ptr_offset = (a_idx.height*a_idx.width) + (b_idx.height*b_idx.width);
  int total_input_elts = mult_ptr_offset + (c_idx.height*c_idx.width);
  float * enclave_data = (float *) malloc(total_input_elts*sizeof(float));
  for(int i = 0; i < total_input_elts; i++){
    enclave_data[i] = data_in[i];
  }
  
  if(verify_frievald(enclave_data, a_idx, b_idx, c_idx)){
    free(enclave_data);
    enclave_data = NULL;
    return 1;
  }
  
  float * activation_buffer_enclave = enclave_data + mult_ptr_offset;
  if(activate(activation_buffer_enclave, c_idx, data_out, matrix_n_out)){
    free(enclave_data);
    enclave_data = NULL;
    return 1;
  }
  
  free(enclave_data);
  enclave_data = NULL;
  return 0;

}
