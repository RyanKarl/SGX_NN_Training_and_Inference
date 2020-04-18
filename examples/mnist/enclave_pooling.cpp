#include <iostream>
#include <vector>
#include <algorithm>
#include <stdio.h>

using namespace std;

void max_pooling(float test_matrix[5][5], int test_h, int test_w, int window_h, int window_w, float pooled_matrix[3][3], int pooled_h, int pooled_w){
    
    int i = 0, j = 0, m = window_h, n = window_w, x = test_h, y = test_w; 
    std::vector<float> v;
    std::vector<float> pool_vector; 

    for(m; m <= x; m++){
        for(n; n <= y; n++){
            for(i; i < m; i++){
                for(j; j < n; j++){
                    v.push_back(*(*(test_matrix + i) + j));
                }
                j = j - window_h;
            }
            
            float max = 0;
            for(int r = 0; r < v.size(); r++){ 
                if(v[r] > max){
                    max = v[r];
                }
            }
            
            pool_vector.push_back(max);
            v.clear();
            i = i - window_w;
            j++;
        }
        n = n - window_h;
        i++;
    }
    
    int z = 0;
    for(int a = 0; a < pooled_h; a++){
       for(int b = 0; b < pooled_w; b++){
       
          pooled_matrix[a][b] = pool_vector[z];
          z++; 
       }
    }

}

int main() 
{ 
    float test_matrix[5][5] = {{1.0,2.0,3.0,4.0,5.0},
                               {6.0,7.0,8.0,9.0,10.0},
                               {11.0,12.0,13.0,14.0,15.0},
                               {16.0,17.0,18.0,19.0,20.0},
                               {21.0,22.0,23.0,24.0,25.0}};       

    float pooled_matrix[3][3];

    max_pooling(test_matrix, 5, 5, 3, 3, pooled_matrix, 3, 3);

    for(int a = 0; a < 3; a++){
       for(int b = 0; b < 3; b++){
          printf("%f\n", pooled_matrix[a][b]);
       }
    }


    return 0; 
}
