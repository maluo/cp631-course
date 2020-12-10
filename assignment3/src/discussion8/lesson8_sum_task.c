/*
** @Author: Ma Luo
** @Date: Nov 12, 2020
** Compile: gcc -fopenmp -O2 lesson8_sum_task.c -o lesson8_sum_task.x
** Run: ./lesson8_sum_task.x
*/

#include "omp.h"
#include "stdlib.h"
#include "stdio.h"

#define N 10

int main(){
    int A[N][N];
    int i = 0, j = 0, sum  =0;

    for(i = 0; i < N; i++){
        for(j = 0; j < N; j++){
            A[i][j] = 1;
        }
    }

#pragma omp parallel
#pragma omp single
{
    for(i=0; i < N; i++){
        for(j=0; j < N; j++){
            sum+=A[i][j];
        }
    }
}
printf ("Total sum of the array is: %d\n",sum);
}