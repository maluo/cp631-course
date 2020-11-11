/* trapezoid.c -- Parallel Trapezoidal Rule
 *
 * Program estimates of the integral from a to b of f(x)
 *    using the trapezoidal rule and n trapezoids.
 * Compile with:  mpicc -O2 trapezoid.c  trapezoid_rule.c  func.c
 */


#include <stdio.h>
#include<math.h>
#include "mpi.h"
#include "omp.h"

#define f(x) x
//1/(1+x*x)

float Simpson_Integral(float lower, float upper, int n, float stepSize)
{

    float integration; /* result of integration  */
    float x;
    int i, k;

    integration = f(lower) + f(upper);

    #pragma omp parallel for collapse(1) schedule(runtime)
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

int main(int argc, char** argv) 
{
    int         my_rank;   /* My process rank           */
    int         p;         /* The number of processes   */
    float       a = 0;   /* Left endpoint             */
    float       b = 12;   /* Right endpoint            */
    int         n = 6;  /* Number of trapezoids      */
    float       h;         /* Trapezoid base length     */
    float       local_a;   /* Left endpoint my process  */
    float       local_b;   /* Right endpoint my process */
    int         local_n;   /* Number of trapezoids for  */
                         /* my calculation            */
    float       integral;  /* Integral over my interval */
    float       total=-1;  /* Total integral            */
    int         source;    /* Process sending integral  */
    int         dest = 0;  /* All messages go to 0      */
    int         tag = 0;

  
    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    h = (b-a)/n;    /* h is the same for all processes */
    local_n = n/p;  /* the number of trapezoids is also same for all processes*/
  
    /* Length of interval of integration on each process = local_n*h. */
    local_a = a + my_rank*local_n*h;
    local_b = local_a + local_n*h;
    integral = Simpson_Integral(local_a, local_b, local_n, h);
  
    /* Sum up the integrals calculated by each process */
    if (my_rank == 0) 
    {

        total = integral; //Give some open mp task here
        for (source = 1; source < p; source++) 
        {
            MPI_Recv(&integral, 1, MPI_FLOAT, source, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("PE %d <- %d,   %f\n", my_rank,source, integral);
            total = total + integral;
	}
    } 
    else 
    {
        printf("PE %d -> %d,   %f\n", my_rank, dest, integral);
        MPI_Send(&integral, 1, MPI_FLOAT, dest, tag, MPI_COMM_WORLD);
    }

    /* Print result */
    if (my_rank == 0) 
    {
        printf("With n = %d sub-integral, our estimate\n", n);
        printf("of the integral from %f to %f = %f\n", a, b, total);
    }
  
    MPI_Finalize();

    return 0;
} 


