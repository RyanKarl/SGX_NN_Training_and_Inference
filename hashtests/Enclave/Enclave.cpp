//Enclave.cpp
//Jonathan S. Takeshita, Ryan Karl, Mark Horeni

//#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>
#include <math.h>
#include <string.h>

#include <vector>
#include <string>

#include "hash.h"

#ifdef NENCLAVE
# include "Enclave.h"
# ifndef OCALLS_H
#  include "../App/ocalls.h"
# endif
#else
# include "Enclave_t.h"
#endif


int enclave_main(unsigned int n, unsigned int s){
  float * data = NULL;
  size_t hash_result = 0;
  data = (float *) malloc(s);
#ifndef NENCLAVE
  sgx_status_t ocall_status;
#endif  
  assert(!(s % sizeof(float)));
  for(unsigned int i = 0; i < n; i++){
    data[0] = hash_result;
    //Start timing
#ifndef NENCLAVE
    ocall_status = start_timing();
    if(ocall_status != SGX_SUCCESS){
      return 1;
    }
#else
    start_timing();
#endif    
    //Do hash
    hash_result = float_hash(data, s / sizeof(float));
    //Stop timing
#ifndef NENCLAVE
    ocall_status = finish_timing();
    if(ocall_status != SGX_SUCCESS){
      return 1;
    }
#else
    finish_timing();
#endif    
  }
  free(data);
  data = NULL;

  return 0;
}
