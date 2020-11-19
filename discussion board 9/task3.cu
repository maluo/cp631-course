/*
Implementation of SAXPY accelerated with CUDA.

A CPU implementation is also included for comparison.

No timing calls or error checks in this version, for clarity.

Compile on graham with:

nvcc -arch=sm_60 -O2 saxpy_cuda.cu 

nvprof ./a.out


*/


#include "cuda.h" /* CUDA runtime API */
#include "cstdio" 

__global__ void prime_gpu(int *vecY, int *vecX, int n) {
    int i;

    i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i<n) {
        int j = 2;
        int prime = 1;
        for (j = 2; j < vecX[i]; j++) {
            if (vecX[i] % j == 0) {
                prime = 0;
                break;
            }
        }
        vecY[i] = prime;
    }
}


int main(int argc, char *argv[]) {
    int *x_host, *y_host;   /* arrays for computation on host*/
    int *x_dev, *y_dev;     /* arrays for computation on device */
    int *y_shadow;          /* host-side copy of device results */

    int n = atoi(argv[1]);
    int k = atoi(argv[2]);

    printf("n = %d\nk = %d\n", n, k);
    
    if (k + n < 0) {
        // make sure stay within in precision
        k = INT_MAX - n;
    }

    size_t memsize;
    int i, blockSize, nBlocks;

    memsize = k * sizeof(int);

    /* allocate arrays on host */

    x_host = (int *)malloc(memsize);
    y_host = (int *)malloc(memsize);
    y_shadow = (int *)malloc(memsize);

    /* allocate arrays on device */

    cudaMalloc((void **) &x_dev, memsize);
    cudaMalloc((void **) &y_dev, memsize);

    /* catch any errors */

    /* initialize arrays on host */

    for ( i = 0; i < k; i++) {
        x_host[i] = n+i;
        y_host[i] = 0;
    }

    /* copy arrays to device memory (synchronous) */

    cudaMemcpy(x_dev, x_host, memsize, cudaMemcpyHostToDevice);
    cudaMemcpy(y_dev, y_host, memsize, cudaMemcpyHostToDevice);

    /* set up device execution configuration */
    blockSize = 512;
    nBlocks = k / blockSize + (k % blockSize > 0);

    /* execute kernel (asynchronous!) */

    prime_gpu<<<nBlocks, blockSize>>>(y_dev, x_dev, k);

    /* retrieve results from device (synchronous) */
    cudaMemcpy(y_shadow, y_dev, memsize, cudaMemcpyDeviceToHost);

    /* guarantee synchronization */
    cudaDeviceSynchronize();

    /* check results */
    for(i=0; i < k; i++) {
        //printf("%d\n", y_shadow[i]);
        if (y_shadow[i] == 1) {
            printf("%d, ", n+i);
        }
    }
    printf("\n");

    /* free memory */
    cudaFree(x_dev);
    cudaFree(y_dev);
    free(x_host);
    free(y_host);
    free(y_shadow);

    return 0;
}


