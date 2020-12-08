#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <stdlib.h>

int **create_Array(int row, int col);//Two dimension
void destroyArray(float **arr);//Two dimension
void destroyPointer(float *arr);//One Dimension
int cal_A(int **arr, int row, int col);
void printMatrix(float **arr, int row, int col);
void printMatrix1D(float *arr, int len);
void init_data(float ***data_ptr, int dim_x, int dim_y, int val);
void init_data_1D(float **data_ptr, int dim_x, float val);
void matrix_mul(int ***data_ptr, int **A, int **B, int row, int col);
void matrix_mul_A(float **data_ptr, float **A, float **B, int row, int col);
void printMatrixFloat(float **arr, int row, int col);
void GET_2D_TO_1D_Float(float **res, float **A, int row, int col);
void GET_1D_TO_2D_Float(float *res, float **A, int row, int col);
void cpuMatMul(float *x, float *y, float *ans, int N, int COL);

void cpuMatMul(float *x, float *y, float *ans, int N, int COL)
{
int i = 0, j = 0, k = 0;
for(i=0;i<N;i++) //row
    {
        for(j=0;j<N;j++) //row
        {
            for(k=0;k<COL;k++) //col
            {
                ans[i*N + j] += (x[i*N+k] * y[k*N+j]);
            }
        }
    }  
}

void GET_1D_TO_2D_Float(float *res, float **A, int row, int col){
    int N = row * col;
    int i = 0;
    int ind_I = 0, ind_J = 0;
    for(i = 0; i< N; i++){
        A[ind_I][ind_J] = res[i];
        ind_J = ind_J + 1;
        if(ind_J == col) {ind_J = 0; ind_I = ind_I + 1;}
    }
}

void GET_2D_TO_1D_Float(float **res, float **A, int row, int col)
{
    float *data;
    data = (float *)malloc(sizeof(float) * row * col);
    int i = 0, j = 0, count = 0;
    for (i = 0; i < row; i++)
    {
        for (j = 0; j < col; j++)
        {
            data[count] = A[i][j];
            count ++;
        }
    }
    *res = data;
}

void matrix_mul(int ***data_ptr, int **A, int **B, int row, int col)
{
    int i, j, k;

    int **data;
    data = (int **)malloc(sizeof(int *) * row);
    for (k = 0; k < row; k++)
    {
        data[k] = (int *)malloc(sizeof(int) * row);
    }

    for (i = 0; i < row; i++)
    {
        for (j = 0; j < row; j++)
        {
            for (k = 0; k < col; k++)
            {
                data[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    *data_ptr = data;
}

void matrix_mul_A(float **data_ptr, float **A, float **B, int row, int col)
{
    int i, j, k;

    for (i = 0; i < row; i++)
    {
        for (j = 0; j < row; j++)
        {
            for (k = 0; k < col; k++)
            {
                data_ptr[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

void printMatrix(float **arr, int row, int col)
{
    int i = 0, j = 0;
    for (i = 0; i < row; i++)
    {
        for (j = 0; j < col; j++)
        {
            printf("%f     ", arr[i][j]);
        }
        printf("\n");
    }
}
void printMatrix1D(float *arr, int len)
{
    int i = 0;
    for (i = 0; i < len; i++)
    {
        printf("%f\n", arr[i]);
    }
}
void printMatrixFloat(float **arr, int row, int col)
{
    int i = 0, j = 0;
    for (i = 0; i < row; i++)
    {
        for (j = 0; j < col; j++)
        {
            printf("%d     ", arr[i][j]);
        }
        printf("\n");
    }
}

int cal_A(int **arr, int row, int col)
{

    int i = 0, j = 0, sum = 0;
    for (i = 0; i <= row; i++)
    {
        for (j = 0; j <= col; j++)
        {
            sum += arr[i][j];
        }
    }

    return sum;
}

void destroyPointer(float *arr)
{
    free(arr);
}

void destroyArray(float **arr)
{
    free(*arr);
    free(arr);
}

void init_data_1D(float **data_ptr, int dim_x, float val){
    float *data;
    int i=0;
    data = (float *)malloc(sizeof(float) * dim_x);
    for (i = 0; i < dim_x; i++){
        data[i] = val;
    }
    *data_ptr = data;
}

void init_data(float ***data_ptr, int dim_x, int dim_y, int val)
{
    int i, j, k;
    float **data;
    data = (float **)malloc(sizeof(float *) * dim_x);
    for (k = 0; k < dim_x; k++)
    {
        data[k] = (float *)malloc(sizeof(float) * dim_y);
    }

    for (i = 0; i < dim_x; i++)
    {
        for (j = 0; j < dim_y; j++)
        {
            data[i][j] = val;
        }
    }
    *data_ptr = data;
}