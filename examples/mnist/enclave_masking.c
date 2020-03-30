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
    

      //ret = tanh(x);      

      //PRINT for DEBUG
      //for (i = 0; i <  r; i++) 
      //    for (j = 0; j < c; j++) 
      //        printf("%d ", *(arr + i*c + j));               
                            
      return 0; 
}
