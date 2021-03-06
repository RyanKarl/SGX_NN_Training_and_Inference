#ifndef ENCLAVE_DEFINES_H
#define ENCLAVE_DEFINES_H

#include <limits.h>
# ifndef CHAR_BIT
# define CHAR_BIT 8
# endif

#include <cstdint>
#include <math.h>

#define FP_TYPE float
#define QUANT_BITS 16

#define FLOAT_RAW_TYPE float
#define SHIFT_T uint64_t
#define USE_FIXED 0

//Code for fixed-point representation, not currently used
inline FLOAT_RAW_TYPE fixed_to_float(FP_TYPE x)
{
#if USE_FIXED
  return ((FLOAT_RAW_TYPE)x / (FLOAT_RAW_TYPE)((SHIFT_T)1 << QUANT_BITS));
#else  
  return x;  
#endif  
}


inline FP_TYPE float_to_fixed(FLOAT_RAW_TYPE input)
{
#if USE_FIXED
  return (FP_TYPE)(round(input * ((SHIFT_T)1 << QUANT_BITS)));
#else
  return input;
#endif  
}


# define NUM_MATRICES 3
# define MAT_DIM 2
# define K_PROBABILITY 2

# define FLOAT_TOLERANCE 1

//Index in row-major order
//i indexes width, j indexes height
//Last arg. should be width
# define INDEX_FLOATMAT(f, i, j, n) (f[(j*n)+(i)])
//Assumes f is a char *
# define INDEX_BITARR(f, i) (( (f[i / CHAR_BIT]) >> (i%CHAR_BIT)) & 1)
# define FLOAT_CMP(a, b) (abs(a - b) >= FLOAT_TOLERANCE)

#define STRUCTURE_BUFLEN 1024


typedef struct {
	int height;
	int width;
} mat_dim_t;

#define FULLY_CONNECTED 0
#define CONVOLUTIONAL 1
#define POOLING 2

#include <string>

typedef struct {
	unsigned int neurons;
	std::string filename;
	int type;
} layer_file_t;

#ifndef M_PI
# define M_PI 3.14159265358979323846
#endif

#define LEARNING_RATE 0.1


#define TASK_ALL 0
#define TASK_FORWARD 1
#define TASK_BACKPROP 2

#define WEIGHTS_SCALE 10


#endif
