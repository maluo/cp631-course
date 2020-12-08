#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <stdlib.h>
#include "matrix_lib.h"

//Fix Number
#define MAX_COL 4
//Could be changed to anything
#define N 4

int main()
{
    int row = N, col = MAX_COL, i, j, k;
    float **A, **B, **MUL;
    float *AH, *BH, *RES;

    init_data(&A, N, MAX_COL, 2);
    init_data(&B, N, MAX_COL, 1);
    init_data_1D(&RES,N*MAX_COL,0);
    init_data(&MUL, N, MAX_COL, 0);

    GET_2D_TO_1D_Float(&AH,A,N,MAX_COL);//Array allocation inside function

    GET_2D_TO_1D_Float(&BH,B,N,MAX_COL);//Array allocation inside function

    cpuMatMul(AH,BH,RES,N,MAX_COL);
    
    GET_1D_TO_2D_Float(RES,MUL,N,MAX_COL);
    printMatrix(MUL, N, MAX_COL);

    destroyArray(A);
    destroyArray(B);
    destroyArray(MUL);
    destroyPointer(RES);
    destroyPointer(AH);
    destroyPointer(BH);

    //Brute force version - not working with CUDA
    
    // init_data(&B, MAX_COL, N, 2);
    // init_data(&MUL, N, N, 0);

    // printf("N - ROWS in A = %d\n", N);

    // printf("A = [\n");
    // printMatrix(A, row, col);
    // printf("]\n");

    // printf("B = [\n");
    // printMatrix(B, col, row);
    // printf("]\n");
    
    // matrix_mul_A(MUL,A,B,row,col);
    
    // printf("MUL = [\n");
    // printMatrix(MUL, row, row);
    // printf("]\n");

    // destroyArray(A);
    // destroyArray(B);
    // destroyArray(MUL);

    return 0;
}
