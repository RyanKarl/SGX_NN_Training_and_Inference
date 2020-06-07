// g++ test_frievald.cpp -Wall -Werror -pedantic -std=c++14
#include <iostream>
#include <stdlib.h> //Need this for rand
#include <assert.h>
/*
#include "./Enclave/Enclave.h"
#include "./App/ocalls.h"
*/

#include "./Enclave/Enclave_Defines.h"

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
    float ** c, int * c_width, int * c_height, const int negate=0, const int alloc_new=1){
  assert(a_width == b_height);
  assert(a_height > 0);
  assert(a_width > 0);
  assert(b_height > 0);
  assert(b_width > 0);

  *c_width = b_width;
  *c_height = a_height;
  if(alloc_new){
    *c = (float *) malloc(sizeof(float)*(*c_width)*(*c_height));
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

//0 is height, 1 is width
int frievald(const float * a, const float * b, const float * c, 
  const int a_width, const int a_height, 
  const int b_width, const int b_height,  
   const int c_width, const int c_height)
{
  //Mult. is defined
  assert(a_width == b_height);
  //Output dims are correct
  assert(c_height == a_height);
  assert(c_width == b_width);
  //Create a random vector r
  //TODO get less randomness
  float * r = (float *) malloc(sizeof(float) * b_width);
  for(int i = 0; i < b_width; i++){
    unsigned char rand_byte;
    rand_bytes(&rand_byte, 1);
    r[i] = (float) (rand_byte & 1);
  }

  /*
  cout << "r: \n";
  print_floatmat(r, b_width, 1);
  */

  float * br = NULL;
  int br_w, br_h;
  float * cr = NULL;
  int cr_w, cr_h;
  
  matrix_multiply(b, b_width, b_height, 
    r, 1, b_width,
    &br, &br_w, &br_h, 0, 1);
  assert(br_h == a_width);
  assert(br_w == 1);

  matrix_multiply(c, c_width, c_height,
    r, 1, c_width,
    &cr, &cr_w, &cr_h, 0, 1);
  assert(cr_h == c_height);
  assert(cr_w == 1);
  
  /*
  cout << "br: \n";
  print_floatmat(br, br_h, br_w);
  
  cout << "cr: \n";
  print_floatmat(cr, cr_h, cr_w);
  */

  free(r);
  r = NULL;

  float * axbr;
  int axbr_w, axbr_h;
  matrix_multiply(a, a_width, a_height, 
    br, br_w, br_h,
    &axbr, &axbr_w, &axbr_h, 0);

  free(br);
  br = NULL;
  /*
  cout << "axbr: \n";
  print_floatmat(axbr, axbr_h, axbr_w);
  */
  for (int i = 0; i < cr_h; i++){
    //cout << "axbr[" << i << "] " << axbr[i] << " cr[" << i << "] " << cr[i] << endl;
    if (FLOAT_CMP(axbr[i], cr[i])){    
        //cout << "axbr " << axbr[i] << " cr " << cr[i] << " i " << i << endl;
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

#define BOUND 5

int main(int argc, char ** argv){

	srand(time(NULL));

	static const int a_w = 784;
	static const int a_h = 1;
	static const int b_w = 500;
	static const int b_h = 784;
	/*
	static const int c_w = 3;
	static const int c_h = 3;
	*/

	float a[a_w*a_h];
	float b[b_w*b_h];
	for(int i = 0; i < a_w * a_h; i++){
		a[i] = (float) i;
		a[i] /= a_w * a_h;
		a[i] /= 100;
	}
	for(int j = 0; j < b_w*b_h; j++){
		b[j] = (float) j;
		b[j] /= b_w*b_h;
		b[j] /= 100;
	}
	/*
	float c[c_w*c_h] = {70, 80, 90,
						158, 184, 210,
						246, 288, 330};
  */


	/*
	float c[c_w*c_h] = {30, 36, 42,
						66, 81, 96,
						102, 126, 150};
						*/

	


	cout << "A" << endl;
	print_floatmat(a, a_h, a_w);
	cout << "B" << endl;
	print_floatmat(b, b_h, b_w);
	

	int c_h_mult, c_w_mult;
	float * c_mult;
	matrix_multiply(a, a_w, a_h, b, b_w, b_h, &c_mult, &c_w_mult, &c_h_mult, 0);
  cout << "C (computer)" << endl;
  print_floatmat(c_mult, c_h_mult, c_w_mult);
  
  int result = frievald(a, b, c_mult, a_w, a_h, b_w, b_h, c_w_mult, c_h_mult);
	cout << "Frievald's algorithm " << (result? "FAILED" : "SUCCEEDED") << endl;
	
	free(c_mult);
	c_mult = NULL;

	return 0;
}
