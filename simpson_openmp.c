// before running, need this command: 
// ulimit -s unlimited

#include <stdio.h>
#include<math.h>
#include "omp.h"

#define f(x) x
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
    float       a = 0;   /* Left endpoint             */
    float       b = 12;   /* Right endpoint            */
    int         n = 6;  /* Number of trapezoids      */
    int num; /*Indexes for threading*/
    float total = 0;  /* Total integral */
    int nthreads, tid;

#pragma omp parallel private(nthreads, tid)
{
    float       h;         /* Trapezoid base length     */
    float       local_a;   /* Left endpoint my process  */
    float       local_b;   /* Right endpoint my process */
    int         local_n;   /* Number of trapezoids for  */
    float       integral = 0;  /* Integral over my interval */

    int numthreads = omp_get_num_threads(); //let's make is as 3
    tid = omp_get_thread_num();
    h = (b-a)/n;    /* h is the same for all processes */
    local_n = n/numthreads;  /* the number of trapezoids is also same for all processes*/

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