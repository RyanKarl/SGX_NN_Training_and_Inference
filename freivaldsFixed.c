//gcc freivaldsFixed.c -std=c99 -g3 -Wall -Werror -pedantic

#include <stdio.h>
#include <stdlib.h>

void print_float_mat(float ** x, int n){
  for(int i = 0; i < n; i++){
    for(int j = 0; j < n; j++){
      printf("%f ", x[i][j]);
    }
    printf("\n");
  }
  printf("\n");
}

void print_float_vec(float * x, int n){
  for(int j = 0; j < n; j++){
      printf("%f ", x[j]);
  }
  printf("\n");
}

//Filler function
int rand_bits(char * r, int n){
  for(int i = 0; i < n; i++){
    r[i] = rand() % 2;
  }
  return 0;
}

char float_cmp(float a, float b){
  //DEBUG
  //printf("%f %f\n", a, b);
  return a != b;
}

int frievald(float ** a, float ** b, float ** c, int n)
{
    // create a random vector r
    char * r = malloc(n);
    rand_bits(r, n);

    //Hope that calloc properly sets bits to 0
    //Calculate br, cr in the same loop
    float * br = calloc(n, sizeof(float));
    float * cr = calloc(n, sizeof(float));
    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            br[i] += b[i][j] * r[j];
            cr[i] += c[i][j] * r[j];
        }
    }
    
    /*
    //DEBUG
    print_float_vec(br, n);
    print_float_vec(cr, n);
    */

    float * axbr = calloc(n, sizeof(float));
    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            axbr[i] += a[i][j] * br[j];
        }
    }
    
    /*
    //DEBUG
    print_float_vec(axbr, n);
    */

    for (int i = 0; i < n; i++){
        //DEBUG
        //printf("Comparing axbr[%d] and cr[%d]\n", i, i);
        if (float_cmp(axbr[i], cr[i])){
            free(axbr);
            free(br);
            free(cr);
            return 1;
        }
    }
    
    free(axbr);
    free(br);
    free(cr);

    return 0;
}

char isMatrixProduct(float ** a, float ** b, float ** c, int n, int k){
    for (int i = 0; i < k; i++){
        if (frievald(a, b, c, n)){
            return 1;
        } // probability of false positive <= 1/(2^k)
    }
    return 0;
}

#define N_DIM 2 

void checkAndPrint(int times)
{

  float ** a = malloc(N_DIM*sizeof(float*));
  float ** b = malloc(N_DIM*sizeof(float*));
  float ** c = malloc(N_DIM*sizeof(float*));
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
    a[i] = malloc(N_DIM*sizeof(float));
    b[i] = malloc(N_DIM*sizeof(float));
    c[i] = malloc(N_DIM*sizeof(float));
    for(int j = 0; j < N_DIM; j++){
      a[i][j] = 1.0;
      b[i][j] = 1.0;
      c[i][j] = 2.0;
      
    }  
  }
  
  /*
  print_float_mat(a, N_DIM);
  print_float_mat(b, N_DIM);
  print_float_mat(c, N_DIM);
  */
  if (isMatrixProduct(a, b, c, N_DIM, times)){
    printf("Verification failed!\n");
  }
  else{
    printf("Verification successful!\n");
  }
}

int main()
{
    srand(5);
    checkAndPrint(20);
    return 0;
}
