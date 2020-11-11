/* @Author: Ma Luo
 * @Date: Nov 11, 2020
 * 
 * Program estimates of the integral from a to b of f(x), could be changed with the defined function
 * 
 * Program running with OpenMP
 *  
 * Compile with:  gcc -fopenmp simpson_openmp.c -o simpson_openmp.x
 * 
 * Assign number of threads going to run with: export OMP_NUM_THREADS = 3, please run with 3 cause we did not make the program that flexbile, this would be based on the 
 *  number of sub-intervals we have. NUMINTVALS / nthreads should be an integer
 * 
 * Run with: ./simpson_openmp.x
 * 
 */

#include <stdio.h>
#include<math.h>
#include "omp.h"

#define f(x) x
#define LOWBOUND 0
#define UPBOUND 12
#define NUMINTVALS 6
//1/(1+x*x)

float Simpson_Integral(float lower, float upper, int n, float stepSize)
{

    float integration; /* result of integration  */
    float x;
    int i, k;

    integration = f(lower) + f(upper);
    for (i = 1; i <= n-1; i++)
    {
        k = lower + i * stepSize;
        if (i % 2 == 0)
        {
            integration = integration + 2 * f(k);
        }
        else
        {
            integration = integration + 4 * f(k);
        }
    }

    return integration = integration * stepSize / 3;
}

int main() {
    float       a = LOWBOUND;   /* Left endpoint */
    float       b = UPBOUND;   /* Right endpoint */
    int         n = NUMINTVALS;  /* Number of sub-intervals */
    int num; /*Indexes for threading*/
    float total = 0;  /* Total integral */
    int nthreads, tid;

#pragma omp parallel private(nthreads, tid)
{
    float       h;
    float       local_a;   /* Left endpoint my process  */
    float       local_b;   /* Right endpoint my process */
    int         local_n;
    float       integral = 0;  /* Integral over my interval */

    int numthreads = omp_get_num_threads(); //let's make is as 3
    tid = omp_get_thread_num();
    h = (b-a)/n;    /* h is the same for all processes */
    local_n = n/numthreads;  

    local_a = a + tid * local_n*h;
    local_b = local_a + local_n*h;
    integral = Simpson_Integral(local_a, local_b, local_n, h);

#pragma omp critical
{
    total = total + integral;
}

}
printf("Total ingegral is: %f \n",total);
}