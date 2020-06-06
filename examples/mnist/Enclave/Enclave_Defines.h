#ifndef ENCLAVE_DEFINES_H
#define ENCLAVE_DEFINES_H

#include <limits.h>
# ifndef CHAR_BIT
# define CHAR_BIT 8
# endif

# define NUM_MATRICES 3
# define MAT_DIM 2
# define K_PROBABILITY 2

# define FLOAT_TOLERANCE 1e-3

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

#define LEARNING_RATE 0.01

#endif
