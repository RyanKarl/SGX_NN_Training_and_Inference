#ifndef ENCLAVE_GLOBALS_H
#define ENCLAVE_GLOBALS_H

#include <limits.h>
# ifndef CHAR_BIT
# define CHAR_BIT 8
# endif

# define NUM_MATRICES 3
# define MAT_DIM 2
# define K_PROBABILITY 2

# define INDEX_FLOATMAT(f, i, j, n) (f[(i*n)+(j)])
//Assumes f is a char *
# define INDEX_BITARR(f, i) (( (f[i / CHAR_BIT]) >> (i%CHAR_BIT)) & 1)
# define FLOAT_CMP(a, b) (a != b)


typedef struct {
	int height;
	int width;
} mat_dim_t;


#endif
