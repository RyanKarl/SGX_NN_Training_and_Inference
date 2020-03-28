//gcc freivaldsFixed.c -std=c99 -g3 -Wall -Werror -pedantic

#include <stdio.h>
#include <stdlib.h>

#include "enclave_functions.h"

#define N_DIM 2 
//Unsafe - memory not being freed
void checkAndPrint(int times)
{

  float * a = malloc(N_DIM*N_DIM*sizeof(float*));
  float * b = malloc(N_DIM*N_DIM*sizeof(float*));
  float * c = malloc(N_DIM*N_DIM*sizeof(float*));
  /*
  //3x3 example
  for(int i = 0; i < N_DIM; i++){
    a[i] = malloc(N_DIM*sizeof(float));
    b[i] = malloc(N_DIM*sizeof(float));
    c[i] = malloc(N_DIM*sizeof(float));
    for(int j = 0; j < N_DIM; j++){
      a[i][j] = 1.0;
      b[i][j] = 1.0;
      if(i == 0 || i == 2 || j == 0){
        c[i][j] = 3.0;
      }
      else{
        c[i][j] = (i==j)? 1.0 : 2.0;
      }
      
    }  
  }
  
  */
  
  
  //2x2 example
  for(int i = 0; i < N_DIM; i++){
    for(int j = 0; j < N_DIM; j++){
      INDEX_FLOATMAT(a, i, j, N_DIM) = 1.0;
      INDEX_FLOATMAT(b, i, j, N_DIM) = 1.0;
      INDEX_FLOATMAT(c, i, j, N_DIM) = 2.0;      
    }  
  }
  
  /*
  print_float_mat(a, N_DIM);
  print_float_mat(b, N_DIM);
  print_float_mat(c, N_DIM);
  */
  
  int dims[N_DIM] = {2,2};
  
  if (frievald(a, b, c, dims, dims, dims)){
    printf("Verification failed!\n");
  }
  else{
    printf("Verification successful!\n");
  }
  
  free(a);
  free(b);
  free(c);
}

int main()
{
    srand(5);
    checkAndPrint(20);
    return 0;
}
