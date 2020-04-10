//Enclave.h
//Jonathan S. Takeshita, Ryan Karl, Mark Horeni

#ifndef ENCLAVE_FUNCTIONS_H
#define ENCLAVE_FUNCTIONS_H

#include "Enclave_Defines.h"

#if defined(__cplusplus)
extern "C" {
#endif

#ifndef NENCLAVE
# include <sgx_trts.h>
# define rand_bytes(r, n_bytes) (sgx_read_rand((unsigned char *) & r, n_bytes) )
#else
# include <stdlib.h> //Need this for rand
# include <assert.h>
inline void rand_bytes(unsigned char * r, size_t n_bytes){
  assert(r);
  for(size_t i = 0; i < n_bytes; i++){
  	r[i] = (unsigned char) rand();
  }
  /*
  for(int i = 0; i < n; i++){
    r[i] = 0;
    for(unsigned int j = 0; j < sizeof(int)*CHAR_BIT; j++){
      r[i] |= (int)((1 & rand()) << j);
    }
  }
  */
  return;
}
#endif

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

#if defined(__cplusplus)
}
#endif

#endif
