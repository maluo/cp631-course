 //Matrix multiplication using shared and non shared kernal

 /*
The program will calculate 
[N*K]*[K*N] matrix, N = R * K
Some of serial functions and operations has been wrapped in matrix_lib.h
nvcc -arch=sm_60 -O2 matrixmul_cuda.cu -o ./matrixmul_cuda.x
nvprof ./matrixmul_cuda.x
*/


 #include <stdio.h>
 #include <math.h>
 #include "matrix_lib.h"
 
 #define TILE_WIDTH 32 //size of smallest block
 #define ROWN 128 // NRow that defines the upper bond

 //non shared
 __global__ void
 MatrixMul( float *Md , float *Nd , float *Pd , const int WIDTH )
 {
 
            // calculate thread id
 
            unsigned int col = TILE_WIDTH*blockIdx.x + threadIdx.x ;
 
            unsigned int row = TILE_WIDTH*blockIdx.y + threadIdx.y ;
 
          for (int k = 0 ; k < TILE_WIDTH ; k++ ) //column size make it as TILE_WIDTH
          {
                   Pd[row*WIDTH + col]+= Md[row * TILE_WIDTH + k ] * Nd[ k * WIDTH + col] ;
                   //ans[i*N + j] += (x[i*COL+k] * y[k*N+j]);
                   //printf("[%d %d %d] => %f => %f * %f\n",row,col,k,Pd[row*WIDTH + col], Md[row * TILE_WIDTH + k ], Nd[ k * WIDTH + col]);
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
 
         for (int m = 0 ; m <WIDTH/TILE_WIDTH ; m++ ) // m indicate number of phase
        {
             //ans[i*N + j] += (x[i*COL+k] * y[k*N+j]);
             Mds[threadIdx.y][threadIdx.x] =  Md[row*TILE_WIDTH + (m*TILE_WIDTH + threadIdx.x)]  ;
             Nds[threadIdx.y][threadIdx.x] =  Nd[ ( m*TILE_WIDTH + threadIdx.y) * WIDTH + col] ;

          __syncthreads() ; // for syncronizeing the threads
 
          // Do for tile
            for ( int k = 0; k < TILE_WIDTH ; k++ ) {
                Pd[row*WIDTH + col]+= Mds[threadIdx.x][k] * Nds[k][threadIdx.y] ;
            }
          __syncthreads() ; // for syncronizeing the threads
      }
 }
 
 // main routine
 int main ()
 {
    const int WIDTH = ROWN ;
    float array1_h[WIDTH][TILE_WIDTH] ,
          array2_h[TILE_WIDTH][WIDTH],
          result_array_h[WIDTH][WIDTH] ,
          M_result_array_h[WIDTH][WIDTH] ;
          
   float *array1_d , *array2_d ,*result_array_d  ,*M_result_array_d ; // device array
   int i , j ;
   int nerror=0; //compute errors and bias

   //Before we start, let run a serial at the mean while
   //To be notifed, all serial operations are wrapped in matrix_lib.h
    float **A, **B, **MUL;
    float *AH, *BH, *RES;

    init_data(&A, WIDTH, TILE_WIDTH, 2);
    init_data(&B, TILE_WIDTH, WIDTH, 2);
    init_data_1D(&RES,WIDTH*WIDTH,0);
    init_data(&MUL, WIDTH, WIDTH, 0);

    GET_2D_TO_1D_Float(&AH,A,WIDTH,TILE_WIDTH);//Array allocation inside function
    GET_2D_TO_1D_Float(&BH,B,TILE_WIDTH,WIDTH);//Array allocation inside function


   //input in host array
   for ( i = 0 ; i<WIDTH ; i++ )
   {
      for (j = 0 ; j<TILE_WIDTH ; j++ )
      {
         array1_h[i][j] = 2 ;
      }
   }

   for ( i = 0 ; i<TILE_WIDTH ; i++ )
   {
      for (j = 0 ; j<WIDTH ; j++ )
      {
         array2_h[i][j] = 2 ;
      }
   }
 
   //create device array cudaMalloc ( (void **)&array_name, sizeofmatrixinbytes) ;
   cudaMalloc((void **) &array1_d , WIDTH*TILE_WIDTH*sizeof (float) ) ;
   cudaMalloc((void **) &array2_d , TILE_WIDTH*WIDTH*sizeof (float) ) ;
 
   //copy host array to device array; cudaMemcpy ( dest , source , WIDTH , direction )
   cudaMemcpy ( array1_d , array1_h , WIDTH*TILE_WIDTH*sizeof (float) , cudaMemcpyHostToDevice ) ;
   cudaMemcpy ( array2_d , array2_h , TILE_WIDTH*WIDTH*sizeof (float) , cudaMemcpyHostToDevice ) ;
 
   //allocating memory for resultent device array
   cudaMalloc((void **) &result_array_d , WIDTH*WIDTH*sizeof (float) ) ;
   cudaMalloc((void **) &M_result_array_d , WIDTH*WIDTH*sizeof (float) ) ;
   // printMatrix1D(array1_d,WIDTH*TILE_WIDTH);
   // printMatrix1D(array2_d,WIDTH*TILE_WIDTH);
 
   //calling kernal
   dim3 dimGrid ( WIDTH/TILE_WIDTH , WIDTH/TILE_WIDTH ,1 ) ;
   dim3 dimBlock( TILE_WIDTH, TILE_WIDTH, 1 ) ;
 
 // Change if 0 to if 1 for running non shared code and make if 0 for shared memory code
 #if 0
 
                 MatrixMul <<<dimGrid,dimBlock>>> ( array1_d , array2_d ,M_result_array_d , WIDTH) ;
 
 #endif
  
 #if 1
 
                MatrixMulSh<<<dimGrid,dimBlock>>> ( array1_d , array2_d ,M_result_array_d , WIDTH) ;
 
 #endif
 
   cpuMatMul(AH,BH,RES,WIDTH,TILE_WIDTH);
   GET_1D_TO_2D_Float(RES,MUL,WIDTH,WIDTH);

   // all gpu function blocked till kernel is working
   //copy back result_array_d to result_array_h
   cudaMemcpy(M_result_array_h , M_result_array_d , WIDTH*WIDTH*sizeof(float) ,
                                     cudaMemcpyDeviceToHost) ;

   /* guarantee synchronization */
   cudaDeviceSynchronize();

   
   //printf the result array
   for ( i = 0 ; i<WIDTH ; i++ )
   {
       for ( j = 0 ; j < WIDTH ; j++ )
      {
         printf ("%f   ",M_result_array_h[i][j] ) ;
         if(M_result_array_h[i][j]!=MUL[i][j]) {nerror=nerror+1;}
      }
   printf ("\n") ;
   }
   printf("\nTest comparison shows %d errors\n",nerror);

   cudaFree(array1_d);
   cudaFree(array2_d);
   cudaFree(result_array_d);
   cudaFree(M_result_array_d);

   destroyArray(A);
   destroyArray(B);
   destroyArray(MUL);
   destroyPointer(RES);
   destroyPointer(AH);
   destroyPointer(BH);
  //system("pause") ;
 }
 