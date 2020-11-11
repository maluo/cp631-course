#include <stdio.h>
#include<math.h>
#include <string.h>
#include <stdlib.h>
#include <sys/time.h>

/*
** @Author: Ma Luo
** @Date : Nov 11, 2020
** @Params: Row and Column number
** @A init: row_index*row + column_index
** Compile: gcc -fopenmp -O2 dynamicmatrix.c -o dynamicmatrix.x
** Running: ./dynamicmatrix.x
**
*/

#define MAX_ROW 10
#define MAX_COL 10

int** create_Array(int row, int col);
void destroyArray(int** arr);
int cal_A(int **arr, int row, int col);
void printMatrix(int **arr);

int main() {

    struct timeval  dtStart, dtEnd;

    int **A, **B;
    A = create_Array(MAX_ROW,MAX_COL);
    B = create_Array(MAX_ROW,MAX_COL);
    int i = 0, j = 0;

    for(i = 0; i < MAX_ROW; i++){
        for(j = 0; j < MAX_COL; j++){
            A[i][j] = i*MAX_ROW + j;
        }
    }

    for(i = 0; i < MAX_ROW; i++){
        for(j = 0; j < MAX_COL; j++){
            B[i][j] = cal_A(A,i,j);
        }
    }

    gettimeofday(&dtStart, NULL);

    printf("A = [\n");
    printMatrix(A);
    printf("]\n");
    printf("B = [\n");
    printMatrix(B);
    printf("]\n");

    gettimeofday(&dtEnd, NULL);

    printf ("Total time = %f seconds\n",
         (double) (dtEnd.tv_usec - dtStart.tv_usec) / 1000000 +
         (double) (dtEnd.tv_sec - dtStart.tv_sec));

    destroyArray(A);
    destroyArray(B);
}

void printMatrix(int **arr){
    int i = 0, j = 0;
    for(i = 0; i < MAX_ROW; i++){
        for(j =0; j < MAX_COL; j++){
            printf("%d     ", arr[i][j]);
        }
        printf("\n");
    }
}

int cal_A(int **arr, int row, int col){
    

    int i = 0, j = 0, sum = 0;
    for(i = 0; i <= row; i++){
        for(j =0; j <= col; j++){
            sum += arr[i][j];
        }
    }

    return sum;
}

void destroyArray(int **arr){
    free(*arr);
    free(arr);
}

int** create_Array(int row, int col){
    int* values = calloc(row*col, sizeof(int));
    int** rows = malloc(col*sizeof(int*));    
    int i = 0;
    for (i=0; i < col; ++i)
    {
        rows[i] = values + i*row;
    }
    return rows;
}