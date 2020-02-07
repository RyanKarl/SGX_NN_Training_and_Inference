#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#define N 3

int frievald(int a[][N], int b[][N], int c[][N])
{
    // create a random vector r
    bool r[N];
    for (int i = 0; i < N; i++){
        r[i] = rand() % 2;
    }

    int br[N] = {0};
    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++){
            br[i] += b[i][j] * r[j];
        }
    }

    int cr[N] = {0};
    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++){
            cr[i] += c[i][j] * r[j];
        }
    }

    int axbr[N] = {0};
    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++){
            axbr[i] += a[i][j] * br[j];
        }
    }

    for (int i = 0; i < N; i++){
        if (axbr[i] - cr[i] != 0){
            return false;
        }
    }

    return true;
}

bool isMatrixProduct(int a[][N], int b[][N], int c[][N], int k){
    for (int i = 0; i < k; i++){
        if (frievald(a, b, c) == false){
            return false;
        } // probability of false positive <= 1/(2^k)
    }
    return true;
}

void checkAndPrint(int times)
{
    int a[N][N] = { { 1, 1, 1 }, { 1, 1, 1 } };
    int b[N][N] = { { 1, 1, 1 }, { 1, 1, 1 } };
    int c[N][N] = { { 2, 2, 2 }, { 2, 2, 2 } };
    int k = 2;
    for (int x = 0; x < times; x++)
    {
        if (isMatrixProduct(a, b, c, k))
            printf("Yes\n");
        else
            printf("No\n");
    }
}

int main()
{
    checkAndPrint(20);
    return 0;
}
