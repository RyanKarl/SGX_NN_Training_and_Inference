#include <stdio.h>
#include <assert.h>
#include <stdlib.h>

# define INDEX_FLOATMAT(f, i, j, n) ((f)[(i*(n))+(j)])

void print_mat(float * data, int width, int height){
  for(int i = 0; i < height; i++){
    for(int j = 0; j < width; j++){
      int idx = (i*width)+j;

      printf("%f (%d) ", INDEX_FLOATMAT(data, i, j, width), idx);
    }
    printf("\n");
  }
}

void print_flat(float * data, int count){
  for(int i = 0; i < count; i++){
    printf("%f ", data[i]);
  }
  printf("\n");
}

void matrix_multiply(float * a, int a_width, int a_height, float * b, int b_width, int b_height, float ** c, int * c_width, int * c_height){
  assert(a_width == b_height);
  assert(a_height > 0);
  assert(a_width > 0);
  assert(b_height > 0);
  assert(b_width > 0);

  *c_width = b_width;
  *c_height = a_height;
  *c = (float *) malloc(sizeof(float)*(*c_width)*(*c_height));

  for(int i = 0; i < a_height; i++){
    //printf("i %d\n", i);
    for(int j = 0; j < b_width; j++){
      //printf("\tj %d\n", j);
      INDEX_FLOATMAT((*c), i, j, (*c_width)) = 0.0f;
      for(int k = 0; k < a_width; k++){
        //printf("\t\tk %d\n", k);
        //printf("\t\t a %f b %f\n", INDEX_FLOATMAT(a, i, k, a_width), INDEX_FLOATMAT(b, k, j, b_width));
        INDEX_FLOATMAT((*c), i, j, (*c_width)) += INDEX_FLOATMAT(a, i, k, a_width)*INDEX_FLOATMAT(b, k, j, b_width);
      }
    }
  }

/*
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
  */
  return;
}



int main(int argc, char ** argv){
  int a_w = 1;
  int a_h = 3;
  int b_w = 2;
  int b_h = a_w;

  float * a = (float *) malloc(sizeof(float) * a_w * a_h);
  float * b = (float *) malloc(sizeof(float) * b_w * b_h);
  
  float counter = 0.0f;
  for(int i = 0; i < a_h; i++){
    for(int j = 0; j < a_w; j++){
      INDEX_FLOATMAT(a, i, j, a_w) = counter;
      counter += 1.0f;
      //printf("\t%f ", INDEX_FLOATMAT(a, i, j, a_h));
    }
    //printf("\n");
  }

  printf("a:\n");
  print_mat(a, a_w, a_h);



  
  counter = 0.0f;
  for(int i = 0; i < b_h; i++){
    for(int j = 0; j < b_w; j++){
      INDEX_FLOATMAT(b, i, j, b_w) = counter;
      int idx = (i*b_w)+j;
      assert(INDEX_FLOATMAT(b, i, j, b_w) == b[idx]);
      //printf("counter %f i %d j %d idx %d val %f\n", counter, i, j, idx, b[idx]);
      //printf("\t%f ", INDEX_FLOATMAT(b, i, j, b_h));
      counter += 1.0f;
    }
    //printf("\n");
  }
  
  printf("\n\nb:\n");
  print_mat(b, b_w, b_h);
  printf("\n");

  float * c;
  int c_w;
  int c_h;
  matrix_multiply(a, a_w, a_h, b, b_w, b_h, &c, &c_w, &c_h);
  assert(c_w == b_w);
  assert(c_h == a_h);
  /*
  for(int i = 0; i < c_h; i++){
    for(int j = 0; j < c_w; j++){
      printf("\t(%d,%d) %f ", i, j, INDEX_FLOATMAT(c, i, j, c_h));
    }
    printf("\n");
  }
  */
  printf("\n\nc:\n");
  print_mat(c, c_w, c_h);
  printf("\n");
  free(c);
  free(a);
  free(b);
  return 0;
}