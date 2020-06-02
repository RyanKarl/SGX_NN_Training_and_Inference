#include <stdio.h>
#include <assert.h>
#include <stdlib.h>

# define INDEX_FLOATMAT(f, i, j, n) (f[(j*n)+(i)])

void print_mat(float * data, int width, int height){
  for(int j = 0; j < height; j++){
    for(int i = 0; i < width; i++){
      int idx = (j*width)+i;

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

void matrix_multiply(const float * a, const int a_width, const int a_height,
    const float * b, const int b_width, const int b_height, 
    float ** c, int * c_width, int * c_height, const int negate=0){
  assert(a_width == b_height);
  assert(a_height > 0);
  assert(a_width > 0);
  assert(b_height > 0);
  assert(b_width > 0);

  *c_width = b_width;
  *c_height = a_height;
  *c = (float *) malloc(sizeof(float)*(*c_width)*(*c_height));
  
  if(!negate){
    for(int i = 0; i < a_height; i++){
      for(int j = 0; j < b_width; j++){
        (*c)[(i*(*c_width))+j] = 0.0f;
        for(int k = 0; k < a_width; k++){
            (*c)[(i*(*c_width))+j] += a[(i*a_width)+k] * b[(k*b_width)+j];
        }
      }
    }
  }
  else{
    for(int i = 0; i < a_height; i++){
      for(int j = 0; j < b_width; j++){
        (*c)[(i*(*c_width))+j] = 0.0f;
        for(int k = 0; k < a_width; k++){
          (*c)[(i*(*c_width))+j] -= a[(i*a_width)+k] * b[(k*b_width)+j];
        }
      }
    }
  }  
  
  return;
}



int main(int argc, char ** argv){
  int a_w = 1;
  int a_h = 3;
  int b_w = 2;
  int b_h = a_w;

  float * a = (float *) malloc(sizeof(float) * a_w * a_h);
  float * b = (float *) malloc(sizeof(float) * b_w * b_h);
  
  float counter = 1.0f;
  for(int i = 0; i < a_w; i++){
    for(int j = 0; j < a_h; j++){
      INDEX_FLOATMAT(a, i, j, a_w) = counter;
      counter += 1.0f;
      //printf("\t%f ", INDEX_FLOATMAT(a, i, j, a_h));
    }
    //printf("\n");
  }

  printf("\na:\n");
  print_mat(a, a_w, a_h);



  
  counter = 1.0f;
  for(int i = 0; i < b_w; i++){
    for(int j = 0; j < b_h; j++){
      INDEX_FLOATMAT(b, i, j, b_w) = counter;
      int idx = (i*b_w)+j;
      //assert(INDEX_FLOATMAT(b, i, j, b_w) == b[idx]);
      //printf("counter %f i %d j %d idx %d val %f\n", counter, i, j, idx, b[idx]);
      //printf("\t%f ", INDEX_FLOATMAT(b, i, j, b_w));
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
  print_flat(c, c_w*c_h);
  
  printf("\n");
  free(c);
  free(a);
  free(b);
  return 0;
}
