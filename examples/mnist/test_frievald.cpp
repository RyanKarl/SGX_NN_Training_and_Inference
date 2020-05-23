// g++ test_frievald.cpp -Wall -Werror -pedantic -std=c++14
#include <iostream>
#include <stdlib.h> //Need this for rand
#include <assert.h>

#include "./Enclave/Enclave.h"
#include "./App/ocalls.h"

using namespace std;


inline void rand_bytes(unsigned char * r, size_t n_bytes){
  assert(r);
  for(size_t i = 0; i < n_bytes; i++){
    r[i] = (unsigned char) rand();
  }
  return;
}


void print_floatmat(const float * fp, int height, int width){
	for(int j = 0; j < height; j++){
		for(int i = 0; i < width; i++){
			cout << fp[(j*width)+i] << ' ';
		}
		cout << endl;
	}
	cout << endl;
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

//TODO change index macro to take width
//0 is height, 1 is width
int frievald(float * a, float * b, float * c, 
  int a_height, int a_width, int b_height, int b_width, int c_height, int c_width)
{
  //Mult. is defined
  assert(a_width == b_height);
  //Output dims are correct
  assert(c_height == a_height);
  assert(c_width == b_width);
  //Create a random vector r
  size_t num_bytes_randarr = (b_height/CHAR_BIT) + (b_height%CHAR_BIT? 1 : 0);
  unsigned char * r = (unsigned char *) calloc(num_bytes_randarr, sizeof(unsigned char));
  if(!r){
    assert(r && "calloc failed");
  }
  rand_bytes(r, num_bytes_randarr);

  cout << "Random numbers: ";
  for(int i = 0; i < b_width; i++){
  	cout << INDEX_BITARR(r, i) << ' ';
  }
  cout << endl;

  //Hope that calloc properly sets bits to 0
  //Calculate br, cr in the same loop
  float * br = (float *) calloc(b_height, sizeof(float));
  float * cr = (float *) calloc(c_height, sizeof(float));
  if(!br){
    assert(br && "malloc failed");
  }
  if(!cr){
    assert(cr && "malloc failed");
  }
  for (int i = 0; i < b_height; i++){
      for (int j = 0; j < b_width; j++){
          br[i] += INDEX_FLOATMAT(b, i, j, b_width) * ((unsigned char)INDEX_BITARR(r, j));
      }
  }

  for (int i = 0; i < c_height; i++){
      for (int j = 0; j < c_width; j++){
          cr[i] += INDEX_FLOATMAT(c, i, j, c_width) * ((unsigned char)INDEX_BITARR(r, j));
      }
  }

  free(r);
  r = NULL;

  float * axbr = (float *) calloc(b_height, sizeof(float));
  if(!axbr){
    assert(axbr && "malloc failed");
  }
  assert(axbr && "Allocating axbr failed!");
  for (int i = 0; i < b_height; i++){
      for (int j = 0; j < b_width; j++){
          axbr[i] += INDEX_FLOATMAT(a, i, j, b_width) * br[j];
      }
  }

  free(br);
  br = NULL;

  cout << "axbr: " << endl;
  print_floatmat(axbr, b_height, 1);

  cout << "cr:" << endl;
  print_floatmat(cr, c_height, 1);

  for (int i = 0; i < c_width; i++){
  	cout << "axbr[" << i << "] " << axbr[i] << " cr[" << i << "] " << cr[i] << endl;
    if (FLOAT_CMP(axbr[i], cr[i])){
        free(axbr);
        free(cr);
        axbr = cr = NULL;
        return 1;
    }
  }

  free(axbr);
  axbr = NULL;
  free(cr);
  cr = NULL;

  return 0;
}

int frievald(float * a, float * b, float * c, 
  int a_height, int a_width, int b_height, int b_width, int c_height, int c_width)
{
  //Mult. is defined
  assert(a_width == b_height);
  //Output dims are correct
  assert(c_height == a_height);
  assert(c_width == b_width);
  //Create a random vector r
  float * r = (float *) malloc(sizeof(float) * b_width);
  for(int i = 0; i < b_width; i++){
  	unsigned char rand_byte;
  	rand_bytes(&rand_byte, 1);
  	r[i] = (float) (rand_byte & 1);
  }

  //Hope that calloc properly sets bits to 0
  //Calculate br, cr in the same loop
  float * br;
  int br_w, br_h;
  float * cr;
  int cr_w, cr_h;
  
  matrix_multiply(b, b_width, b_height, 
  	r, 1, b_width,
    &br, &br_w, &br_h, 0);
  assert(br_h == a_width);
  assert(br_w == 1);

  matrix_multiply(c, c_width, c_height,
  	r, 1, b_width,
  	&cr, &cr_w, &cr_h, 0);
  assert(cr_h == b_width);
  assert(cr_w == 1);

  free(r);
  r = NULL;

  float * axbr;
  int axbr_w, axbr_h;
  assert(axbr && "Allocating axbr failed!");
  matrix_multiply(a, a_width, a_height, 
  	br, br_w, br_h,
    &axbr, &axbr_w, &axbr_h, 0);

  free(br);
  br = NULL;

  cout << "axbr: " << endl;
  print_floatmat(axbr, b_height, 1);

  cout << "cr:" << endl;
  print_floatmat(cr, c_height, 1);

  for (int i = 0; i < c_width; i++){
  	cout << "axbr[" << i << "] " << axbr[i] << " cr[" << i << "] " << cr[i] << endl;
    if (FLOAT_CMP(axbr[i], cr[i])){
        free(axbr);
        free(cr);
        axbr = cr = NULL;
        return 1;
    }
  }

  free(axbr);
  axbr = NULL;
  free(cr);
  cr = NULL;

  return 0;
}

int main(int argc, char ** argv){

	srand(5);

	static const int a_w = 4;
	static const int a_h = 3;
	static const int b_w = 3;
	static const int b_h = 4;
	static const int c_w = 3;
	static const int c_h = 3;

	float a[a_w*a_h];
	float b[b_w*b_h];
	for(int i = 0; i < a_w * a_h; i++){
		a[i] = 1+i;
	}
	for(int j = 0; j < b_w*b_h; j++){
		b[j] = 1+j;
	}
	float c[c_w*c_h] = {70, 80, 90,
						158, 184, 210,
						246, 288, 330};



	/*
	float c[c_w*c_h] = {30, 36, 42,
						66, 81, 96,
						102, 126, 150};
						*/

	int result = frievald2(a, b, c, a_h, a_w, b_h, b_w, c_h, c_w);


	cout << "A" << endl;
	print_floatmat(a, a_h, a_w);
	cout << "B" << endl;
	print_floatmat(b, b_h, b_w);
	cout << "C (human)" << endl;
	print_floatmat(c, c_h, c_w);

	int c_h_mult, c_w_mult;
	float * c_mult;
	matrix_multiply(a, a_w, a_h, b, b_w, b_h, &c_mult, &c_w_mult, &c_h_mult, 0);
    cout << "C (computer)" << endl;
    print_floatmat(c_mult, c_h_mult, c_w_mult);
    free(c_mult);

	cout << "Frievald's algorithm " << (result? "FAILED" : "SUCCEEDED") << endl;

	return 0;
}