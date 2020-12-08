//Matrix multiplication using shared and non shared kernal

 /*
nvcc -arch=sm_60 -O2 testcuda.cu -o ./testcuda.x
nvprof ./testcuda.x
*/


#include <stdio.h>
#include <math.h>
#include "matrix_lib.h"

#define TILE_WIDTH 2
#define ROWN 4

/*matrix multiplication kernels*/

 
void matrix_mul_A( float **MUL, float **A, float **B, int row, int col) {
   int i, j, k;
    for (i = 0; i < row; i++)
    {
        for (j = 0; j < row; j++)
        {
            for (k = 0; k < col; k++)
            {
               MUL[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

//non shared
__global__ void
MatrixMul( float *Md , float *Nd , float *Pd , const int WIDTH )
{
           // calculate thread id
         unsigned int col = TILE_WIDTH*blockIdx.x + threadIdx.x ;

         unsigned int row = TILE_WIDTH*blockIdx.y + threadIdx.y ;

         for (int k = 0 ; k<WIDTH ; k++ )
         {
                  Pd[row*WIDTH + col]+= Md[row * WIDTH + k ] * Nd[ k * WIDTH + col] ;
         }
}

// shared
__global__ void
MatrixMulSh( float *Md , float *Nd , float *Pd , const int WIDTH )
{
        //Taking shared array to break the MAtrix in Tile widht and fatch them in that array per ele
          __shared__ float Mds [TILE_WIDTH][TILE_WIDTH] ;
           __shared__ float Nds [TILE_WIDTH][TILE_WIDTH] ;

         // calculate thread id
          unsigned int col = TILE_WIDTH*blockIdx.x + threadIdx.x ;
          unsigned int row = TILE_WIDTH*blockIdx.y + threadIdx.y ;

        for (int m = 0 ; m<WIDTH/TILE_WIDTH ; m++ ) // m indicate number of phase
       {
            Mds[threadIdx.y][threadIdx.x] =  Md[row*WIDTH + (m*TILE_WIDTH + threadIdx.x)]  ;
            Nds[threadIdx.y][threadIdx.x] =  Nd[ ( m*TILE_WIDTH + threadIdx.y) * WIDTH + col] ;
         __syncthreads() ; // for syncronizeing the threads

         // Do for tile
           for ( int k = 0; k<TILE_WIDTH ; k++ )
                       Pd[row*WIDTH + col]+= Mds[threadIdx.x][k] * Nds[k][threadIdx.y] ;
         __syncthreads() ; // for syncronizeing the threads

     }
}

// main routine
int main ()
{
   const int WIDTH = ROWN ;
   int **array1_h, **array2_h, **result_array_h, **M_result_array_h;

   int **array1_d , **array2_d , **result_array_d  , **M_result_array_d ; // device array
   int i , j ;

   size_t memsize;
   size_t memsize_result;

  //input in host array
  init_data(&array1_h, WIDTH, TILE_WIDTH, 2);
  init_data(&array2_h, TILE_WIDTH, WIDTH, 2);
  init_data(&result_array_h, WIDTH, WIDTH, 0);
  init_data(&M_result_array_h, WIDTH, WIDTH, 0);

  memsize = WIDTH*TILE_WIDTH*sizeof (int);
  memsize_result = WIDTH*WIDTH*sizeof (int);

  //create device array cudaMalloc ( (void **)&array_name, sizeofmatrixinbytes) ;

  cudaMalloc((void **) &array1_d ,  memsize) ;
  cudaMalloc((void **) &array2_d , memsize) ;

  //copy host array to device array; cudaMemcpy ( dest , source , WIDTH , direction )
  cudaMemcpy ( array1_d , array1_h , memsize , cudaMemcpyHostToDevice );
  cudaMemcpy ( array2_d , array2_h , memsize , cudaMemcpyHostToDevice );

  //allocating memory for resultent device array
  cudaMalloc((void **) &result_array_d , memsize_result);
  cudaMalloc((void **) &M_result_array_d , memsize_result);

  //calling kernal

  dim3 dimGrid ( WIDTH/TILE_WIDTH , TILE_WIDTH ,1 );

  dim3 dimBlock( TILE_WIDTH, TILE_WIDTH, 1 );

// Change if 0 to if 1 for running non shared code and make if 0 for shared memory code
#if 0

                MatrixMul <<<dimGrid,dimBlock>>> ( array1_d , array2_d ,M_result_array_d , WIDTH) ;

#endif
 
#if 0

               MatrixMulSh<<<dimGrid,dimBlock>>> ( array1_d , array2_d ,M_result_array_d , WIDTH) ;

#endif

#if 1
               //host version, test with CPU code
               matrix_mul_A(result_array_h,array1_h,array2_h,WIDTH,TILE_WIDTH);
               printMatrix(result_array_h, WIDTH, WIDTH);       
#endif

  // all gpu function blocked till kernel is working
  //copy back result_array_d to result_array_h

  cudaMemcpy(M_result_array_h , M_result_array_d , memsize_result ,
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

  cudaFree(array1_d);
  cudaFree(array2_d);
  cudaFree(result_array_d);
  cudaFree(M_result_array_d);

  destroyArray(array1_h);
  destroyArray(array2_h);
  destroyArray(result_array_h);
  destroyArray(M_result_array_h);

 //system("pause") ;
}
