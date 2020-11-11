/*
* To compile this code, first enable a newer version of gcc with
* scl enable devtoolset-7 bash - This is needed to support advanced OpenMP features
*/

/*
** @Author: Ma Luo
** @Date : Nov 11, 2020
** @Params: Row and Column number
** @A init: row_index*row + column_index
** Compile: gcc -fopenmp -O2 dymatrixmptask.c -o dymatrixmptask.x
** Running: ./dymatrixmptask.x
**
*/

#include "omp.h"
#include "stdlib.h"
#include "stdio.h"
#include <sys/time.h>

#define N 10
#define BS 2

void matmul_tasks (int **A, int **B);
int** create_Array(int row, int col);
void destroyArray(int** arr);
int cal_A(int **arr, int row, int col);
void printMatrix(int **arr);

int main ()
{
    struct timeval  dtStart, dtEnd;

    int **A, **B;
    int i,j;

    A = create_Array(N,N);
    B = create_Array(N,N);

    for(i = 0; i < N; i++){
        for(j = 0; j < N; j++){
            A[i][j] = i*N + j;
        }
    }

    gettimeofday(&dtStart, NULL);

    printf("A = [\n");
    printMatrix(A);
    printf("]\n");

    matmul_tasks (A, B);

    printf("B = [\n");
    printMatrix(B);
    printf("]\n");

    gettimeofday(&dtEnd, NULL);

    printf ("Total time = %f seconds\n",
         (double) (dtEnd.tv_usec - dtStart.tv_usec) / 1000000 +
         (double) (dtEnd.tv_sec - dtStart.tv_sec));

    destroyArray(A);
    destroyArray(B);

  return 0;
}

void matmul_tasks (int **A, int **B)
{
   int i, j, k, ii, jj, kk;
#pragma omp parallel
#pragma omp single
{

   for (i = 0; i < N; i+=BS)
     for (j = 0; j < N; j+=BS){
#pragma omp task firstprivate(i,j) private(ii, jj)
{
            for (ii = i; ii < i+BS; ii++ )
              for (jj = j; jj < j+BS; jj++ )
                  B[ii][jj] = cal_A(A,ii,jj);
//printf("task worked on by thread %d \n",omp_get_thread_num());
}//end of task
}//end of for loop
}//end of single
}

void printMatrix(int **arr){
    int i = 0, j = 0;
    for(i = 0; i < N; i++){
        for(j =0; j < N; j++){
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