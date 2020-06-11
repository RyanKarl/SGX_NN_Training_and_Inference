#ifndef MATRIX_H
#define MATRIX_H
#include <cmath>
#include <cassert>
#include <algorithm>

#include "Enclave_Defines.h"


void matrix_add(const FP_TYPE * a, const FP_TYPE * b, int elts, FP_TYPE * result){
  for(int i = 0; i < elts; i++){
    result[i] = a[i] + b[i];
  }
}

void matrix_sub(const FP_TYPE * a, const FP_TYPE * b, int elts, FP_TYPE * result){
  for(int i = 0; i < elts; i++){
    result[i] = a[i] - b[i];
  }
}

//Allocates memory
void matrix_multiply(const FP_TYPE * a, const int a_width, const int a_height,
    const FP_TYPE * b, const int b_width, const int b_height, 
    FP_TYPE ** c, int * c_width, int * c_height, const int negate=0, const int alloc_new=1){
  assert(a_width == b_height);
  assert(a_height > 0);
  assert(a_width > 0);
  assert(b_height > 0);
  assert(b_width > 0);

  *c_width = b_width;
  *c_height = a_height;
  if(alloc_new){
    *c = (FP_TYPE *) malloc(sizeof(FP_TYPE)*(*c_width)*(*c_height));
  }
  
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

void sub_from_ones(FP_TYPE * a, const int total_elts){
  for(int i = 0; i < total_elts; i++){
    a[i] = 1.0 - a[i];
  }
  return;
}

int activate(FP_TYPE * data, int height, int width){
  if(height <= 0 || width <= 0){
    return 1;
  }
  for(int i = 0; i < height * width; i++){
    data[i] = tanh(data[i]);
  }
  return 0;
}

int activate_derivative(FP_TYPE * data, int total_elts){
  for(int i = 0; i < total_elts; i++){
    FP_TYPE tmp = tanh(data[i]);
    data[i] = 1.0f - (tmp*tmp);
  }
  return 0;
}



FP_TYPE * transpose(const FP_TYPE * x, const int width, const int height){
  FP_TYPE * ret = (FP_TYPE *) malloc(sizeof(FP_TYPE) * width * height);
  for(int i = 0; i < width; i++){
    for(int j = 0; j < height; j++){
      INDEX_FLOATMAT(ret, j, i, height) = INDEX_FLOATMAT(x, i, j, width);
    }
  }
  return ret;
}
/*
#include <iostream>
using std::cout;
using std::endl;
void print_floatmat(const FP_TYPE * fp, int width, int height){
  for(int j = 0; j < height; j++){
    for(int i = 0; i < width; i++){
      cout << fp[(j*width)+i] << ' ';
    }
    cout << endl;
  }
  cout << endl;
}
*/

//TODO rewrite to cut down on the loops
//https://stats.stackexchange.com/questions/338285/how-does-the-subtraction-of-the-logit-maximum-improve-learning
#define FP_LARGE_T double
void softmax(FP_TYPE * x, const int width, const int height){
#ifdef DEBUG
  assert(total_elts > 0);
#endif  

  //Max. elt. of a row
  FP_TYPE * max_elts = (FP_TYPE *) malloc(sizeof(FP_TYPE)*height);
  for(int i = 0; i < height; i++){
    for(int j = 0; j < width; j++){
      max_elts[i] = !j ? x[(i*width)] : std::max(x[(i*width)+j], max_elts[i]);
    }
  }
  //print_floatmat(max_elts, 1, height);
  //Subtract the max. from each element of the rows
  for(int i = 0; i < height; i++){
    for(int j = 0; j < width; j++){
      x[(i*width)+j] -= max_elts[i];
    }
  }
  //print_floatmat(x, width, height);
  free(max_elts);
  max_elts = NULL;
  FP_TYPE * sums = (FP_TYPE *) calloc(height, sizeof(FP_TYPE));
  //Exponentiate, get sums of rows
  for(int i = 0; i < height; i++){
    for(int j = 0; j < width; j++){
      sums[i] += (x[(i*width)+j] = exp(x[(i*width)+j]));
    }
  }
  //print_floatmat(x, width, height);
  //print_floatmat(sums, 1, height);
  //Divide
  for(int i = 0; i < height; i++){
    for(int j = 0; j < width; j++){
      x[(i*width)+j] /= sums[i];
    }
  }
  //print_floatmat(x, width, height);
  free(sums);
  sums = NULL;
  return;

  /*
  //Get maximum element of a column
  FP_TYPE * max_elts = (FP_TYPE *) malloc(sizeof(FP_TYPE)*height);
  for(int i = 0; i < height; i++){
    for(int j = 0; j < width; j++){
      max_elts[j] = !i ? x[i] : std::max(x[(j*width)+i], max_elts[i]);
    }
  }
  print_floatmat(max_elts, width, 1);
  //Calculate x - x's col. max.
  for(int i = 0; i < width; i++){
    for(int j = 0; j < height; j++){
       x[(j*height)+i] -= max_elts[i];
    }
  }
  print_floatmat(x, width, height);
  free(max_elts);
  max_elts = NULL;
  FP_TYPE * x_tmp = (FP_TYPE *) malloc(sizeof(FP_TYPE) * width*height);
  //Exponentiate the matrix - hope this fits in FP_TYPE32!
  FP_TYPE * sums = (FP_TYPE *) calloc(height, sizeof(FP_TYPE));
  for(int i = 0; i < height; i++){
    for(int j = 0; j < width; j++){
      x_tmp[(j*height)+i] = exp(x[(j*width)+i]);
      sums[i] += x_tmp[(j*width)+i];
    }   
  }
  print_floatmat(x_tmp, width, height);
  print_floatmat(sums, width, 1);

  for(int i = 0; i < height; i++){
    for(int j = 0; j < width; j++){
      x[(j*height)+i] = x_tmp[(j*height)+i]/sums[i];
    }   
  }
  print_floatmat(x, width, height);

  free(sums);
  sums = NULL;
  free(x_tmp);
  return;
  */
}

//https://stats.stackexchange.com/questions/215521/how-to-find-derivative-of-softmax-function-for-the-purpose-of-gradient-descent/328095
//Returns a pointer to dynamically freed memory
void softmax_derivative(const FP_TYPE * y, const int n, FP_TYPE * y_squared){
  //First, create the identity matrix
  //FP_TYPE * y_squared;
  int y_sq_h, y_sq_w;
  matrix_multiply(y, 1, n, y, n, 1, (FP_TYPE **) &y_squared, &y_sq_w, &y_sq_h, 0, 0); 
  assert(y_sq_w == n);
  assert(y_sq_h == n); 
  for(int i = 0; i < n; i++){
    for(int j = 0; j < n; j++){
      INDEX_FLOATMAT(y_squared, i, j, n) = (i != j)?
       -INDEX_FLOATMAT(y_squared, i, j, n) : 
       y[j] - INDEX_FLOATMAT(y_squared, i, j, n);
    }
  }
  return;
}

/*
FP_TYPE * softmax_derivative(FP_TYPE * s, 
  const int num_vectors, const int vector_height){
  FP_TYPE * ret = (FP_TYPE *) calloc(num_vectors*vector_height*vector_height, sizeof(FP_TYPE));
  //Index ret as a list of square matrices
  for(int i = 0; i < num_vectors; i++){
    for(int j = 0; j < vector_height; j++){
      ret[(i*vector_height*vector_height)+(j*num_vectors)+j] = s[(i*vector_height) + j];
    }
  }
  //ret now holds the diagonal embedding in a 3d matrix

}
*/

FP_TYPE * transform(const FP_TYPE * y, const FP_TYPE * term, const int total_elts){
  FP_TYPE * ret = (FP_TYPE *) malloc(sizeof(FP_TYPE)*total_elts);
  for(int i = 0; i < total_elts; i++){
    FP_TYPE tmp = tanh(y[i] + term[i]);
    ret[i] = 1.0f - (tmp*tmp);
  }
  return ret;
}

void transform_and_mult(const FP_TYPE * y, const FP_TYPE * g, const FP_TYPE * term, FP_TYPE * ret, const int total_elts){
  for(int i = 0; i < total_elts; i++){
    FP_TYPE tmp = tanh(g[i] + term[i]);
    ret[i] = y[i] * (1.0f - (tmp*tmp));
  }
  return;
}

/*
FP_TYPE * transform(const FP_TYPE * x, const FP_TYPE * term, const int width, const int height, const int use_softmax){
  FP_TYPE * difference = (FP_TYPE *) malloc(sizeof(FP_TYPE) * width * height);
  matrix_sub(x, term, width*height, difference);
  if(!use_softmax){
    activate(difference, width, height);
  }
  else{
    softmax(difference, width*height);
  }
  FP_TYPE * squared;
  int p_w, p_h;
  matrix_multiply(difference, width, height, difference, width, height, &squared, &p_w, &p_h, 0);
  free(difference);
  difference = NULL;
  sub_from_ones(squared, width*height);
  return squared;
}
*/

int argmax(FP_TYPE * data, int total_elts){
  int idx = 0;
  FP_TYPE max_data = data[0];
  for(int i = 0; i < total_elts; i++){
    if(data[i] > max_data){
      max_data = data[i];
      idx = i;
    }
  }
  return idx;
}

int nan_idx(const FP_TYPE * data, const int num_elts){
  for(int i = 0; i < num_elts; i++){
    if(isnan(data[i])){
      return i;
    }
  }
  return -1;
}

int bounds_check(const FP_TYPE * data, const int num_elts, const FP_TYPE min, const FP_TYPE max){
  for(int i = 0; i < num_elts; i++){
    if(data[i] < min || data[i] >= max){
      return 1;
    }
  }
  return 0;
}


#endif