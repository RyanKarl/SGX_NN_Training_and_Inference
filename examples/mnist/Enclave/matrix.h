#ifndef MATRIX_H
#define MATRIX_H
#include <cmath>

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

//https://stats.stackexchange.com/questions/338285/how-does-the-subtraction-of-the-logit-maximum-improve-learning
#define FP_LARGE_T double
void softmax(float * x, const int total_elts){
#ifdef DEBUG
  assert(total_elts > 0);
#endif  
  //Get maximum element
  float max_elt = x[0];
  for(int i = 0; i < total_elts; i++){
    if(max_elt > x[i]){
      max_elt = x[i];
    }
  }
  //Calculate x - x.max
  for(int i = 0; i < total_elts; i++){
    x[i] -= max_elt;
  }
  FP_LARGE_T * x_tmp = malloc(sizeof(FP_LARGE_T) * total_elts);
  //Exponentiate the matrix - hope this fits in float32!
  FP_LARGE_T sum = 0;
  for(int i = 0; i < total_elts; i++){
    x_tmp[i] = exp(x[i]);
    sum += x_tmp[i];
#ifdef DEBUG
    assert(x_tmp[i] > 0);
    assert(x_tmp[i] < 1);
#endif    
  }

  for(int i = 0; i < total_elts; i++){
    x[i] = x_tmp[i]/sum;
  }
  free(x_tmp);
  return;
}

//https://stats.stackexchange.com/questions/215521/how-to-find-derivative-of-softmax-function-for-the-purpose-of-gradient-descent/328095
void softmax_derivative(float * y, const int n){
  //First, create the identity matrix
  float * y_squared;
  int y_sq_h, y_sq_w;
  matrix_multiply(y, 1, n,
    y, n, 1, 
    &y_squared, &y_sq_w, &y_sq_h, 0);
#ifdef DEBUG  
  assert(y_sq_w == n);
  assert(y_sq_h == n);
#endif  
  for(int i = 0; i < n; i++){
    for(int j = 0; j < n; j++){
      INDEX_FLOATMAT(y, i, j, n) = (i == j)?
       -INDEX_FLOATMAT(y_squared, i, j, n) : 
       INDEX_FLOATMAT(y, i, j, n) - INDEX_FLOATMAT(y_squared, i, j, n);
    }
  }
  free(y_squared);
  return;
}



#endif