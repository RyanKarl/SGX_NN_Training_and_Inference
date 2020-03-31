//enclave_functions.h
//Jonathan S. Takeshita, Ryan Karl, Mark Horeni

#ifndef ENCLAVE_FUNCTIONS_H
#define ENCLAVE_FUNCTIONS_H

#include <stdio.h>
#include <assert.h>

#ifndef CHAR_BIT
#define CHAR_BIT 8
#endif

#define NUM_MATRICES 3
#define MAT_DIM 2
#define K_PROBABILITY 2

#define INDEX_FLOATMAT(f, i, j, n) (f[(i*n)+(j)])
#define INDEX_BITARR(f, i) (( (f[i / (sizeof(int)*CHAR_BIT)]) >> (i%(sizeof(int)*CHAR_BIT))) & 1)
#define FLOAT_CMP(a, b) (a != b)

//Filler function
void rand_bits(int * r, int n){
  for(int i = 0; i < n; i++){
    r[i] = 0;
    for(int j = 0; j < sizeof(int)*CHAR_BIT; j++){
      r[i] |= (int)((1 & rand()) << j);
    }
  }
  return;
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
  
  //DEBUG
  fprintf(stdout, "Allocating r with %d elements\n", b_idx[1]);
  fflush(stdout);
  
  int * r = calloc(b_idx[1], sizeof(int));
  //DEBUG
  assert(r && "Allocating r failed!");
  rand_bits(r, b_idx[1]);
  
  //DEBUG
  fprintf(stdout, "Rand. bits: ");
  fflush(stdout);
  for(int i = 0; i < b_idx[1]; i++){
    fprintf(stdout, "%d ", r[i]);
    fflush(stdout);
  }
  fprintf(stdout, "\n");

  //Hope that calloc properly sets bits to 0
  //Calculate br, cr in the same loop
  float * br = calloc(b_idx[1], sizeof(float));
  float * cr = calloc(c_idx[1], sizeof(float));
  assert(br && "Allocating br failed!");
  assert(cr && "Allocating cr failed!");
  for (int i = 0; i < b_idx[0]; i++){
      for (int j = 0; j < b_idx[1]; j++){
          br[i] += INDEX_FLOATMAT(b, i, j, b_idx[0]) * ((int)INDEX_BITARR(r, j));
      }
  }

  for (int i = 0; i < c_idx[0]; i++){
      for (int j = 0; j < c_idx[1]; j++){
          cr[i] += INDEX_FLOATMAT(c, i, j, c_idx[0]) * ((int)INDEX_BITARR(r, j));
      }
  }


  //DEBUG
  fprintf(stdout, "Freeing r\n");
  fflush(stdout);
  assert(r && "Pointer r is OK");
  free(r);
  
  
  /*
  //DEBUG
  print_float_vec(br, n);
  print_float_vec(cr, n);
  */

  float * axbr = calloc(b_idx[0], sizeof(float));
  assert(axbr && "Allocating axbr failed!");
  for (int i = 0; i < b_idx[0]; i++){
      for (int j = 0; j < b_idx[1]; j++){
          axbr[i] += INDEX_FLOATMAT(a, i, j, b_idx[0]) * br[j];
      }
  }

  //DEBUG
  fprintf(stdout, "Freeing br\n");
  fflush(stdout);
  free(br);
  
  /*
  //DEBUG
  print_float_vec(axbr, n);
  */

  for (int i = 0; i < c_idx[1]; i++){
      //DEBUG
      //printf("Comparing axbr[%d] and cr[%d]\n", i, i);
      if (FLOAT_CMP(axbr[i], cr[i])){
          free(axbr);
          free(cr);
          return 1;
      }
  }
  
  //DEBUG
  fprintf(stdout, "Freeing axbr\n");
  fflush(stdout);
  
  free(axbr);
  
  
  //DEBUG
  fprintf(stdout, "Freeing cr\n");
  fflush(stdout);
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
  float ** data_out, int * matrix_n_out, int * alloc_new_data){
  //Use the below if things are done in-place
  //data_outshape must have DATA_DIMENSIONS elements
  //Set this to 0 if malloc is used for *data_out
  *alloc_new_data = 0;
  for(unsigned int i = 0; i < MAT_DIM; i++){
    matrix_n_out[i] = matrix_n[i];
  }

  *data_out = data_in;

  for(int j = 0; j < matrix_n[0]*matrix_n[1]*NUM_MATRICES; j++){
    (*data_out)[j] *= 2.0;
  }  

  return 0;
}

#endif
