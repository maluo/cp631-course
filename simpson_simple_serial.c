#include<stdio.h>
#include<math.h>
#include <sys/time.h>

/* Define function here */
#define f(x) x
//1/(1+x*x)

int main()
{
 struct timeval  dtStart, dtEnd;

 float lower, upper, integration=0.0, stepSize, k;
 int i, subInterval;
 /* Input */
 printf("Enter lower limit of integration: ");
 scanf("%f", &lower);
 printf("Enter upper limit of integration: ");
 scanf("%f", &upper);
 printf("Enter number of sub intervals: ");
 scanf("%d", &subInterval);


 gettimeofday(&dtStart, NULL);
 /* Calculation */
 /* Finding step size */
 stepSize = (upper - lower)/subInterval;  

 /* Finding Integration Value */
 integration = f(lower) + f(upper);
 for(i=1; i<= subInterval-1; i++)
 {
  k = lower + i*stepSize;
  if(i%2==0)
  {
   integration = integration + 2 * f(k);
  }
  else
  {
   integration = integration + 4 * f(k);
  }
 }
 integration = integration * stepSize/3;

 gettimeofday(&dtEnd, NULL);

 printf("\nRequired value of integration is: %.3f", integration);
 printf ("\nTotal time = %f seconds\n",
         (double) (dtEnd.tv_usec - dtStart.tv_usec) / 1000000 +
         (double) (dtEnd.tv_sec - dtStart.tv_sec));
 return 0;
}