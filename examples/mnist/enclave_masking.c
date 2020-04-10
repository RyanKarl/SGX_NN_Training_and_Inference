#include <stdio.h>
#include <stdlib.h> 
#include <math.h>
#include <time.h>

/*
int maxpool(int *matrix, int m_height_width, int k_height_width, int stride)
{
    int *max_matrix = (int *)malloc((m_height_width - k_height_width) / stride * sizeof(int));
    
    int i, j, max = 0;
        for (i = 0; i <  k_height_width; i++)
        {
            for (j = 0; j < k_height_width; j++)
                if ((*(matrix + i*k_height_width + j)) > max);
                {
                    max = *(matrix + i*k_height_width + j);
                }  

        }
    return max;
}*/

/*
void mask(int*  matrix, int m_height, int m_width, int r)
{
    int i, j, count = 0;
        for (i = 0; i <  m_height; i++)
            for (j = 0; j < m_width; j++)
                *(matrix + i*m_width + j) = *(matrix + i*m_width + j) + r;
}
*/


int main() 
{ 

    srand(time(NULL));   // Initialization, should only be called once.
    int r = rand() % 10;      // Returns a pseudo-random integer between 0 and RAND_MAX.
    

    int h = 4, w = 4;
    //matrix is primary input 
    int *matrix = (int *)malloc(h * w * sizeof(int)); 
              
    int i, j, count = 0; 
    for (i = 0; i <  h; i++) 
        for (j = 0; j < w; j++) 
            *(matrix + i*w + j) = count++; 

    //printf("Test");
    for (i = 0; i <  h; i++)
        for (j = 0; j < w; j++)
            printf("%d ", *(matrix + i*w + j));

    
    for (i = 0; i <  h; i++)
        for (j = 0; j < w; j++)
            *(matrix + i*w + j) = *(matrix + i*w + j) + r;

    for (i = 0; i <  h; i++)
        for (j = 0; j < w; j++)
            printf("%d ", *(matrix + i*w + j));
      
      //maxpool(*matrix, h, 2, 1);


	'''
def pooling(mat,ksize,method='max',pad=False):
    '''Non-overlapping pooling on 2D or 3D data.

    <mat>: ndarray, input array to pool.
    <ksize>: tuple of 2, kernel size in (ky, kx).
    <method>: str, 'max for max-pooling, 
                   'mean' for mean-pooling.
    <pad>: bool, pad <mat> or not. If no pad, output has size
           n//f, n being <mat> size, f being kernel size.
           if pad, output has size ceil(n/f).

    Return <result>: pooled matrix.
    '''

    m, n = mat.shape[:2]
    ky,kx=ksize

    _ceil=lambda x,y: int(numpy.ceil(x/float(y)))

    if pad:
        ny=_ceil(m,ky)
        nx=_ceil(n,kx)
        size=(ny*ky, nx*kx)+mat.shape[2:]
        mat_pad=numpy.full(size,numpy.nan)
        mat_pad[:m,:n,...]=mat
    else:
        ny=m//ky
        nx=n//kx
        mat_pad=mat[:ny*ky, :nx*kx, ...]

    new_shape=(ny,ky,nx,kx)+mat.shape[2:]

    if method=='max':
        result=numpy.nanmax(mat_pad.reshape(new_shape),axis=(1,3))
    else:
        result=numpy.nanmean(mat_pad.reshape(new_shape),axis=(1,3))

    return result
	'''
    

      //ret = tanh(x);      

      //PRINT for DEBUG
      //for (i = 0; i <  r; i++) 
      //    for (j = 0; j < c; j++) 
      //        printf("%d ", *(arr + i*c + j));               
                            
      return 0; 
}
