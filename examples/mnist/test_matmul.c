#include <stdio.h>
#include <assert.h>
#include <stdlib.h>

# define INDEX_FLOATMAT(f, i, j, n) (f[(i*n)+(j)])

void matrix_multiply(float * a, int a_width, int a_height, float * b, int b_width, int b_height, float ** c, int * c_width, int * c_height){
  assert(a_width == b_height);
  assert(a_height > 0);
  assert(a_width > 0);
  assert(b_height > 0);
  assert(b_width > 0);

  int m = a_height;
  int n = a_width;
  int p = b_width;


  *c_width = p;
  *c_height = m;
  *c = (float *) malloc(sizeof(float)*(*c_width)*(*c_height));

  printf("m %d n %d p %d\n\n", m, n, p);

  for(int i = 0; i < m; i++){
    printf("i %d\n", i);
    for(int j = 0; j < p; j++){
      printf("\tj %d\n", j);
      int c_idx = (i*(*c_width))+j;
      (*c)[c_idx] = 0.0f;

      for(int k = 0; k < n; k++){
        printf("\t\tk %d\n", k);
        int a_idx = (i*a_width)+k;
        int b_idx = (k*b_width)+j;
        //printf("Indices: a %d b %d\n", a_idx, b_idx);

        float a_val = a[a_idx];
        float b_val = b[b_idx];
        //printf("\tValues: a %f b %f\n", a_val, b_val);
        (*c)[c_idx] += a_val*b_val;
      }
    }
  }
  return;
}

int main(int argc, char ** argv){
  int a_w = 2;
  int a_h = 3;
  int b_w = 3;
  int b_h = a_w;

  float * a = (float *) malloc(sizeof(float) * a_w * a_h);
  float * b = (float *) malloc(sizeof(float) * b_w * b_h);
  printf("a:\n");
  for(int i = 0; i < a_h; i++){
    for(int j = 0; j < a_w; j++){
      INDEX_FLOATMAT(a, j, i, a_h) = 1.0*((j*a_h)+i);
      printf("\t%f ", a[(j*a_h)+i]);
    }
    printf("\n");
  }
  printf("\n\nb:\n");
  for(int i = 0; i < b_h; i++){
    for(int j = 0; j < b_w; j++){
      INDEX_FLOATMAT(b, j, i, b_h) = 1.0*((j*b_h)+i);
      printf("\t%f ", b[(j*b_h)+i]);
    }
    printf("\n");
  }
  printf("\n");
  float * c;
  int c_w;
  int c_h;
  matrix_multiply(a, a_w, a_h, b, b_w, b_h, &c, &c_w, &c_h);
  assert(c_w == b_w);
  assert(c_h == a_h);
  printf("\n\nc:\n");
  for(int i = 0; i < c_h; i++){
    for(int j = 0; j < c_w; j++){
      //INDEX_FLOATMAT(c, j, i, a_h) = 1.0*((j*c_h)+i);
      printf("\t%f ", c[(j*c_h)+i]);
    }
    printf("\n");
  }
  printf("\n");
  free(c);
  free(a);
  free(b);
  return 0;
}