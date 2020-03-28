//enclave_functions.h
//Jonathan S. Takeshita, Ryan Karl, Mark Horeni

#ifndef ENCLAVE_FUNCTIONS_H
#define ENCLAVE_FUNCTIONS_H

#include <assert.h>

#ifndef CHAR_BIT
#define CHAR_BIT 8
#endif

#define NUM_MATRICES 3
#define MAT_DIM 2
#define K_PROBABILITY 2

#define INDEX_FLOATMAT(f, i, j, n) (f[(i*n)+(j)])
#define INDEX_BITARR(f, i) (( (f[i / (sizeof(int)*CHAR_BIT)]) >> (i%(sizeof(int)*CHAR_BIT))) & 1)


//Filler function - TODO pack fields
void rand_bits(int * r, int n){
  for(int i = 0; i < n; i++){
    r[i] = 0;
    for(int j = 0; j < sizeof(int)*CHAR_BIT; j++){
      r[i] |= (1 & rand()) << j;
    }
  }
  return;
}

int float_cmp(float a, float b){
  //DEBUG
  //printf("%f %f\n", a, b);
  return a != b;
}

//a, b, c are flattened 2d arrays
int frievald(float * a, float * b, float * c, 
  int a_idx[MAT_DIM], int b_idx[MAT_DIM], int c_idx[MAT_DIM])
{
  //Mult. is defined
  assert(a_idx[1] == b_idx[0]);
  //Output dims are correct
  assert(c_idx[0] == a_idx[0]);
  assert(c_idx[1] == b_idx[1]);
  //Create a random vector r
  int * r = malloc(b_idx[1]*sizeof(int));
  rand_bits(r, b_idx[1]);

  //Hope that calloc properly sets bits to 0
  //Calculate br, cr in the same loop
  float * br = calloc(b_idx[1], sizeof(float));
  float * cr = calloc(c_idx[1], sizeof(float));
  for (int i = 0; i < b_idx[0]; i++){
      for (int j = 0; j < b_idx[1]; j++){
          br[i] += INDEX_FLOATMAT(b, i, j, b_idx[0]) * INDEX_BITARR(r, j);
      }
  }

  for (int i = 0; i < c_idx[0]; i++){
      for (int j = 0; j < c_idx[1]; j++){
          cr[i] += INDEX_FLOATMAT(c, i, j, c_idx[0]) * INDEX_BITARR(r, j);
      }
  }

  free(r);
  
  /*
  //DEBUG
  print_float_vec(br, n);
  print_float_vec(cr, n);
  */

  float * axbr = calloc(b_idx[0], sizeof(float));
  for (int i = 0; i < b_idx[0]; i++){
      for (int j = 0; j < b_idx[1]; j++){
          axbr[i] += INDEX_FLOATMAT(a, i, j, b_idx[0]) * br[j];
      }
  }

  free(br);
  
  /*
  //DEBUG
  print_float_vec(axbr, n);
  */

  for (int i = 0; i < c_idx[1]; i++){
      //DEBUG
      //printf("Comparing axbr[%d] and cr[%d]\n", i, i);
      if (float_cmp(axbr[i], cr[i])){
          free(axbr);
          free(cr);
          return 1;
      }
  }
  
  free(axbr);
  free(cr);

  return 0;
}

//Return 1 if verification fails, 0 if successful
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

//Return 1 if activation fails, 0 if successful
int activate(float * data_in, int matrix_n[MAT_DIM], 
  float ** data_out, int ** matrix_n_out){
  //Use the below if things are done in-place
  //data_outshape must have DATA_DIMENSIONS elements
  for(unsigned int i = 0; i < MAT_DIM; i++){
    (*matrix_n_out)[i] = matrix_n[i];
  }
  *data_out = data_in;

  for(int j = 0; j < matrix_n[0]*matrix_n[1]*NUM_MATRICES; j++){
    (*data_out)[j] *= 2.0;
  }  

  return 0;
}

#endif
