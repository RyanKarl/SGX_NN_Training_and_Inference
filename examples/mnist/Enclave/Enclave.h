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
# define rand_bytes(r, n_bytes) (sgx_read_rand((unsigned char *) r, n_bytes) )
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
  mat_dim_t a_idx, mat_dim_t b_idx, mat_dim_t c_idx);

//Return 1 if verification fails, 0 if successful
int verify_frievald(float * data, mat_dim_t a_idx, mat_dim_t b_idx, mat_dim_t c_idx);

//Return 1 if activation fails, 0 if successful
//Assume data_out is already properly initialized
int activate(float * data_in, mat_dim_t matrix_n, 
  float * data_out, mat_dim_t matrix_n_out); //Reference or value?

//Trust that data_in and data_out have the correct size
//Buffers must be allocated outside the enclave!
int verify_and_activate(float * data_in, mat_dim_t a_idx, mat_dim_t b_idx, mat_dim_t c_idx, float * data_out, mat_dim_t matrix_n_out);

#if defined(__cplusplus)
}
#endif

#endif
