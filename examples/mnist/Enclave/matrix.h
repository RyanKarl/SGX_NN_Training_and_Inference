#include "Enclave_Defines.h"


void matrix_add(const float * a, const float * b, int elts, float * result){
  for(int i = 0; i < elts; i++){
    result[i] = a[i] + b[i];
  }
}

void matrix_sub(const float * a, const float * b, int elts, float * result){
  for(int i = 0; i < elts; i++){
    result[i] = a[i] - b[i];
  }
}

//Allocates memory
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
  }
  else{
    for(int i = 0; i < a_height; i++){
      //printf("i %d\n", i);
      for(int j = 0; j < b_width; j++){
        //printf("\tj %d\n", j);
        INDEX_FLOATMAT((*c), i, j, (*c_width)) = 0.0f;
        for(int k = 0; k < a_width; k++){
          //printf("\t\tk %d\n", k);
          //printf("\t\t a %f b %f\n", INDEX_FLOATMAT(a, i, k, a_width), INDEX_FLOATMAT(b, k, j, b_width));
          INDEX_FLOATMAT((*c), i, j, (*c_width)) -= INDEX_FLOATMAT(a, i, k, a_width)*INDEX_FLOATMAT(b, k, j, b_width);
        }
      }
    }
  }
  
  return;
}

void sub_from_ones(float * a, const int total_elts){
  for(int i = 0; i < total_elts; i++){
    a[i] = 1.0 - a[i];
  }
  return;
}

int activate(float * data, int height, int width){
  if(height <= 0 || width <= 0){
    return 1;
  }
  for(int i = 0; i < height * width; i++){
    data[i] = tanh(data[i]);
  }
  return 0;
}

float * transform(const float * x, const float * term, const int width, const int height){
  float * difference = (float *) malloc(sizeof(float) * width * height);
  matrix_sub(x, term, width*height, difference);
  activate(difference, width, height);
  float * squared;
  int p_w, p_h;
  matrix_multiply(difference, width, height, difference, width, height, &squared, &p_w, &p_h, 0);
  free(difference);
  difference = NULL;
  sub_from_ones(squared, width*height);
  return squared;
}

float * transpose(const float * x, const int width, const int height){
  float * ret = (float *) malloc(sizeof(float) * width * height);
  for(int w_idx = 0; w_idx < width; w_idx++){
    for(int h_idx = 0; h_idx < height; h_idx++){
      INDEX_FLOATMAT(ret, h_idx, w_idx, height) = INDEX_FLOATMAT(x, w_idx, h_idx, width);
    }
  }
  return ret;
}