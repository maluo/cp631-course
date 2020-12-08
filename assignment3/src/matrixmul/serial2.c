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
    printMatrix(A,N,MAX_COL);
    init_data(&B, MAX_COL, N, 1);
    printMatrix(B,MAX_COL,N);
    init_data_1D(&RES,N*MAX_COL,0);
    init_data(&MUL, N, MAX_COL, 0);

    GET_2D_TO_1D_Float(&AH,A,N,MAX_COL);//Array allocation inside function
    printMatrix1D(AH,N*MAX_COL);

    GET_2D_TO_1D_Float(&BH,B,MAX_COL,N);//Array allocation inside function
    printMatrix1D(BH,N*MAX_COL);

    // cpuMatMul(AH,BH,RES,N,MAX_COL);
    // GET_1D_TO_2D_Float(RES,MUL,N,MAX_COL);
    // printMatrix(MUL, N, MAX_COL);

    destroyArray(A);
    destroyArray(B);
    destroyArray(MUL);
    destroyPointer(RES);
    destroyPointer(AH);
    destroyPointer(BH);

    return 0;
}
