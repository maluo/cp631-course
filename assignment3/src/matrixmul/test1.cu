 //Matrix multiplication using shared and non shared kernal

 /*
nvcc -arch=sm_60 -O2 test1.cu -o ./test1.x
nvprof ./test1.x
*/


#include <stdio.h>
#include <math.h>
#include "matrix_lib.h"

#define TILE_WIDTH 2
#define ROWN 6

//Need a convert function here from 2D to 1D for the host version, then we can compare, make it in the lib function

//Serial
void cpuMatMul(float *x, float *y, float *ans, const int N)
{
for(int i=0;i<N;i++) //row
    {
        for(int j=0;j<N;j++) //row
        {
            for(int k=0;k<N;k++) //col
            {
                ans[i*N + j] += (x[i*N+k] * y[k*N+j]);
            }
        }
    }  
}

//non shared
__global__ void
MatrixMul( float *x , float *y , float *ans , const int N )
{
         unsigned int col = TILE_WIDTH*blockIdx.x + threadIdx.x ;

         unsigned int row = TILE_WIDTH*blockIdx.y + threadIdx.y ;

         for (int k = 0 ; k<N ; k++ )
         {
                ans[row*N + col]+= x[row * N + k ] * y[ k * N + col] ;
         }
}

// shared
__global__ void
MatrixMulSh( float *x , float *y , float *ans , const int N )
{
        __shared__ float xs [TILE_WIDTH][TILE_WIDTH] ;
        __shared__ float ys [TILE_WIDTH][TILE_WIDTH] ;
         
        unsigned int col = TILE_WIDTH*blockIdx.x + threadIdx.x ;
        unsigned int row = TILE_WIDTH*blockIdx.y + threadIdx.y ;

       for (int m = 0 ; m < N/TILE_WIDTH ; m++ ) // m indicate number of phase
       {
        xs[threadIdx.y][threadIdx.x] =  x[row*N + (m*TILE_WIDTH + threadIdx.x)]  ;
        ys[threadIdx.y][threadIdx.x] =  y[ (m*TILE_WIDTH + threadIdx.y) * N + col] ;
        __syncthreads() ; // for syncronizeing the threads
        // Do for tile
           for ( int k = 0; k<TILE_WIDTH ; k++ )
                ans[row*N + col]+= xs[threadIdx.x][k] * ys[k][threadIdx.y] ;
         __syncthreads() ; // for syncronizeing the threads
     }
}

// main routine
int main ()
{
   const int WIDTH = ROWN ;
   float array1_h[WIDTH][WIDTH] ,
         array2_h[WIDTH][WIDTH],
         result_array_h[WIDTH][WIDTH] ,
         M_result_array_h[WIDTH][WIDTH] ;
  float *h_array1, *h_array2, *h_result_array;         
  float *array1_d , *array2_d ,*result_array_d  ,*M_result_array_d ; // device array
  int i , j ;
  //input in host array
  for ( i = 0 ; i<WIDTH ; i++ )
  {
     for (j = 0 ; j<WIDTH ; j++ )
     {
        array1_h[i][j] = 1 ;
        array2_h[i][j] = 2 ;
     }
  }

  //create device array cudaMalloc ( (void **)&array_name, sizeofmatrixinbytes) ;
  cudaMalloc((void **) &array1_d , WIDTH*WIDTH*sizeof (int) ) ;
  cudaMalloc((void **) &array2_d , WIDTH*WIDTH*sizeof (int) ) ;

  //copy host array to device array; cudaMemcpy ( dest , source , WIDTH , direction )
  cudaMemcpy ( array1_d , array1_h , WIDTH*WIDTH*sizeof (int) , cudaMemcpyHostToDevice ) ;
  cudaMemcpy ( array2_d , array2_h , WIDTH*WIDTH*sizeof (int) , cudaMemcpyHostToDevice ) ;

  //allocating memory for resultent device array
  cudaMalloc((void **) &result_array_d , WIDTH*WIDTH*sizeof (int) );
  cudaMalloc((void **) &M_result_array_d , WIDTH*WIDTH*sizeof (int) );

  //calling kernal
  dim3 dimGrid ( WIDTH/TILE_WIDTH , WIDTH/TILE_WIDTH ,1 ) ;
  dim3 dimBlock( TILE_WIDTH, TILE_WIDTH, 1 ) ;

// Change if 0 to if 1 for running non shared code and make if 0 for shared memory code
#if 1

              MatrixMul <<<dimGrid,dimBlock>>> ( array1_d , array2_d ,M_result_array_d , WIDTH) ;

#endif
 
#if 0

              MatrixMulSh<<<dimGrid,dimBlock>>> ( array1_d , array2_d ,M_result_array_d , WIDTH) ;

#endif
      
  //cpuMatMul(h_array1, h_array2, h_result_array , WIDTH) ; //compare host and device versions
  //Need some convention here

  // all gpu function blocked till kernel is working
  //copy back result_array_d to result_array_h
  cudaMemcpy(M_result_array_h , M_result_array_d , WIDTH*WIDTH*sizeof(int) ,
                                    cudaMemcpyDeviceToHost) ;
         
  //printf the result array
  for ( i = 0 ; i<WIDTH ; i++ )
  {
     for ( j = 0 ; j < WIDTH ; j++ )
     {
        printf ("%f   ",M_result_array_h[i][j] ) ;
     }
     printf ("\n") ;
  }


 //system("pause") ;
}
