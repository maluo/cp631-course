#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <stdlib.h>

int **create_Array(int row, int col);
void destroyArray(int **arr);
int cal_A(int **arr, int row, int col);
void printMatrix(int **arr, int row, int col);
void init_data(int ***data_ptr, int dim_x, int dim_y, int val);
void matrix_mul(int ***data_ptr, int **A, int **B, int row, int col);

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

void printMatrix(int **arr, int row, int col)
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

void destroyArray(int **arr)
{
    free(*arr);
    free(arr);
}

int **create_Array(int row, int col)
{
    int *values = calloc(row * col, sizeof(int));
    int **rows = malloc(col * sizeof(int *));
    int i = 0;
    for (i = 0; i < col; ++i)
    {
        rows[i] = values + i * row;
    }
    return rows;
}

void init_data(int ***data_ptr, int dim_x, int dim_y, int val)
{
    int i, j, k;
    int **data;
    data = (int **)malloc(sizeof(int *) * dim_x);
    for (k = 0; k < dim_x; k++)
    {
        data[k] = (int *)malloc(sizeof(int) * dim_y);
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