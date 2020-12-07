#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <stdlib.h>
#include "matrix_lib.h"

//Fix Number
#define MAX_COL 2
//Could be changed to anything
#define N 4

int main()
{
    int row = N, col = MAX_COL, i, j, k;
    int **A, **B, **MUL;

    init_data(&A, N, MAX_COL, 2);
    init_data(&B, MAX_COL, N, 2);
    init_data(&MUL, N, N, 0);

    printf("N - ROWS in A = %d\n", N);

    for (i = 0; i < row; i++)
    {
        for (j = 0; j < col; j++)
        {
            A[i][j] = 2;
        }
    }

    printf("A = [\n");
    printMatrix(A, row, col);
    printf("]\n");

    for (i = 0; i < col; i++)
    {
        for (j = 0; j < row; j++)
        {
            B[i][j] = 2;
        }
    }

    printf("B = [\n");
    printMatrix(B, col, row);
    printf("]\n");

    printf("multiply of the matrix=\n");
    for (i = 0; i < row; i++)
    {
        for (j = 0; j < row; j++)
        {
            for (k = 0; k < col; k++)
            {
                MUL[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    
    printf("MUL = [\n");
    printMatrix(MUL, row, row);
    printf("]\n");

    destroyArray(A);
    destroyArray(B);
    destroyArray(MUL);

    return 0;
}
