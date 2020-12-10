/*
** @Author: Ma Luo
** @Date: Nov 12, 2020
** Compile: gcc -fopenmp -O2 monitor.c -o monitor.x
** Run: OMP_NUM_THREADS=4 ./monitor.x
*/


#include "omp.h"
#include "stdlib.h"
#include "stdio.h"

#define N 10

int main(){
omp_lock_t locks[N];
int nthreads, thread_id;
int i = 0;

for(i = 0; i < N, i++;){
    omp_init_lock(&locks[i]);
}

#pragma omp parallel 
//private(nthreads, thread_id) 
{
    int i = 0;
    thread_id = omp_get_thread_num();

    if (thread_id ==0){
        while(1){
            for(i = 1; i < N; i++){
                if(omp_test_lock(&locks[i])){
                    printf("Thread inactive, activating %d\n", thread_id);
                    omp_unset_lock(&locks[i]);
                }
            }
        }
    } 
    else{
        while(1){
            if(omp_test_lock(&locks[thread_id])){
                printf("Thread is currently active: %d\n", thread_id);
                sleep(2);
                omp_unset_lock(&locks[thread_id]);
                printf("Thread is inactive: %d\n",thread_id);
            }
        }
    }
}


}