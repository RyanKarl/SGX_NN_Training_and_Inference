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
  int a_height, int a_width, int b_height, int b_width, int c_height, int c_width)
{
  //Mult. is defined
  assert(a_width == b_height);
  //Output dims are correct
  assert(c_height == a_height);
  assert(c_width == b_width);
  //Create a random vector r
  size_t num_bytes_randarr = (b_width/CHAR_BIT) + (b_width%CHAR_BIT? 1 : 0);
  unsigned char * r = (unsigned char *) calloc(num_bytes_randarr, sizeof(unsigned char));
  if(!r){
    assert(r && "calloc failed");
  }
  rand_bytes(r, num_bytes_randarr);

  //Hope that calloc properly sets bits to 0
  //Calculate br, cr in the same loop
  float * br = (float *) calloc(b_width, sizeof(float));
  float * cr = (float *) calloc(c_width, sizeof(float));
  if(!br){
    assert(br && "malloc failed");
  }
  if(!cr){
    assert(cr && "malloc failed");
  }
  for (int i = 0; i < b_height; i++){
      for (int j = 0; j < b_width; j++){
          br[i] += INDEX_FLOATMAT(b, i, j, b_height) * ((unsigned char)INDEX_BITARR(r, j));
      }
  }

  for (int i = 0; i < c_height; i++){
      for (int j = 0; j < c_width; j++){
          cr[i] += INDEX_FLOATMAT(c, i, j, c_height) * ((unsigned char)INDEX_BITARR(r, j));
      }
  }

  free(r);
  r = NULL;

  float * axbr = (float *) calloc(b_height, sizeof(float));
  if(!axbr){
    assert(axbr && "malloc failed");
  }
  assert(axbr && "Allocating axbr failed!");
  for (int i = 0; i < b_height; i++){
      for (int j = 0; j < b_width; j++){
          axbr[i] += INDEX_FLOATMAT(a, i, j, b_height) * br[j];
      }
  }

  free(br);
  br = NULL;

  for (int i = 0; i < c_width; i++){
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

int verify_frievald(float * data, int a_height, int a_width, int b_height, int b_width, int c_height, int c_width){
  float * matrix_offsets[NUM_MATRICES];
  matrix_offsets[0] = data;
  matrix_offsets[1] = matrix_offsets[0] + (a_height*a_width);
  matrix_offsets[2] = matrix_offsets[1] + (b_height*b_width);
  for(unsigned int j = 0; j < K_PROBABILITY; j++){
  	if(frievald(matrix_offsets[0], matrix_offsets[1], matrix_offsets[2], a_height, a_width, b_height, b_width, c_height, c_width)){
  		return 1;
  	}
  }
  return 0;
}

int activate(float * data_in, int in_height, int in_width, 
  float * data_out, int out_height, int out_width){
  //Use the below if things are done in-place
  //data_outshape must have DATA_DIMENSIONS elements
  /*
  for(unsigned int i = 0; i < MAT_DIM; i++){
    matrix_n_out[i] = matrix_n[i];
  }
  */

  //Using tanh as the activation function
  assert(out_height == in_height && out_width == in_width);
  for(int j = 0; j < out_height*out_width; j++){
    data_out[j] = tanh(data_in[j]);
  }  

  return 0;
}

int verify_and_activate(float * data_in, int a_height, int a_width, int b_height, int b_width, int c_height, int c_width,
 float * data_out, int out_height, int out_width){
  //Copy data to enclave space
  //Validate data here
  if(a_height < 0 || a_width < 0 ||
     b_height < 0 || b_width < 0 ||
     c_height < 0 || c_width < 0 ||
     out_height < 0 || out_width < 0){
    return 1;
  }
  int mult_ptr_offset = (a_height*a_width) + (b_height*b_width);
  int total_input_elts = mult_ptr_offset + (c_height*c_width);
  float * enclave_data = (float *) malloc(total_input_elts*sizeof(float));
  for(int i = 0; i < total_input_elts; i++){
    enclave_data[i] = data_in[i];
  }
  
  if(verify_frievald(enclave_data, a_height, a_width, b_height, b_width, c_height, c_width)){
    free(enclave_data);
    enclave_data = NULL;
    return 1;
  }
  
  float * activation_buffer_enclave = enclave_data + mult_ptr_offset;
  if(activate(activation_buffer_enclave, c_height, c_width, data_out, out_height, out_width)){
    free(enclave_data);
    enclave_data = NULL;
    return 1;
  }
  
  free(enclave_data);
  enclave_data = NULL;
  return 0;

}
