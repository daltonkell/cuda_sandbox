#include <iostream>
#include <cuda_runtime.h> // CUDA routines prefixed with cuda_
#include <stdio.h>
#include <time.h> // time() for timing functions

// cuda_runtime.h includes
// stdlib.h -> rand(), RAND_MAX, malloc, calloc EXIT_FAILURE, EXIT_SUCCESS, exit() (among others)

// Add two equally-sized vectors (1D) element-wise.
// Results are contained in the third vector, C.
// Parameters:
//   A, B, C: float *
// Returns:
//   void
__global__ void
vecAddKernel(const float *A, const float *B, float *C)
{

    // built-in variables blockDim, blockIdx, threadIdx
    // allow us to assign one thread per operation
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    C[i] = A[i] + B[i];

    // TODO throw exc?

}

// This function utilizes device intrinsics to compute the addition
__global__ void
vecAddIntrinsicKernel(const float *A, const float *B, float *C)
{

    // thread ID, one for each operation
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    // round-nearest
    C[i] = __fadd_rn(A[i], B[i]);
}

// function to get time in nanoseconds
long get_nanos(void)
{
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (long)ts.tv_sec * 1000000000L + ts.tv_nsec;
}

int main()
{

    // TODO make dynamic via CLI input

    // variable to be used for checking results of CUDA routines
    cudaError_t cudaErrResult = cudaSuccess;

    int nElements = 25000; // number of elements per vector
    size_t vecSize = nElements * sizeof(float); // size in bytes per vector

    // pointers for host and device vectors
    float *hostA = NULL;
    float *hostB = NULL;
    float *hostC = NULL;
    float *devA = NULL;
    float *devB = NULL;
    float *devC = NULL;

    // time variables for use in timing functions
    long nSecondsStart;
    long nSecondsEnd;

    // host memory allocation; allocate all to 0
    hostA = (float*)calloc(nElements, vecSize);
    hostB = (float*)calloc(nElements, vecSize);
    hostC = (float*)calloc(nElements, vecSize);

    // check that none of our alloc'd ptrs are null
    if (hostA == NULL || hostB == NULL || hostC == NULL)
    {
        std::cout << "ERROR: COULD NOT ALLOCATE HOST MEMORY!" << std::endl;
        exit(EXIT_FAILURE);
    }

    // CUDA device memory allocation; we'll need to cast our ptrs
    // to ptr to void ptr; NOTE how the cast is performed on the address
    // of the float ptr

    // allocate for device array A ...
    cudaErrResult = cudaMalloc((void **)&devA, vecSize);
    if (cudaErrResult != cudaSuccess)
    {
        std::cout << "CUDA ERROR: " << cudaErrResult
                  << " COULD NOT ALLOCATE DEVICE MEMORY FOR ARRAY A!"
                  << std::endl
                  << "Error: " << cudaGetErrorString(cudaErrResult)
                  << std::endl;
        exit(EXIT_FAILURE);
    }

    // ... for device array B ...
    cudaErrResult = cudaMalloc((void **)&devB, vecSize);
    if (cudaErrResult != cudaSuccess)
    {
        std::cout << "CUDA ERROR: " << cudaErrResult
                  << " COULD NOT ALLOCATE DEVICE MEMORY FOR ARRAY B!"
                  << std::endl
                  << "Error: " << cudaGetErrorString(cudaErrResult)
                  << std::endl;
        exit(EXIT_FAILURE);
    }

    // ... and finally device array C
    cudaErrResult = cudaMalloc((void **)&devC, vecSize);
    if (cudaErrResult != cudaSuccess)
    {
        std::cout << "CUDA ERROR: " << cudaErrResult
                  << " COULD NOT ALLOCATE DEVICE MEMORY FOR ARRAY C!"
                  << std::endl
                  << "Error: " << cudaGetErrorString(cudaErrResult)
                  << std::endl;
        exit(EXIT_FAILURE);
    }
    // initlialize arrays on host; make arrays of random numbers
    for (int i=0; i<nElements; ++i)
    {
        hostA[i] = rand()/(float)RAND_MAX;
        hostB[i] = rand()/(float)RAND_MAX;
    }

    printf("Just sanity checking host arrays...\n");
    printf("hostA first element: %8.6f\n", hostA[0]);
    printf("hostA last element: %8.6f\n", hostA[nElements-1]);
    printf("hostB first element: %8.6f\n", hostB[0]);
    printf("hostB last element: %8.6f\n", hostB[nElements-1]);

    // copy host memory input arrays to device memory input arrays
    // NOTE that cudaMemcpyHostToDevice is enum
    cudaErrResult = cudaMemcpy(devA, hostA, vecSize, cudaMemcpyHostToDevice);
    if (cudaErrResult != cudaSuccess)
    {
        std::cout << "FAILURE COPYING hostA TO devA!"
                  << std::endl
                  << "Error: " << cudaGetErrorString(cudaErrResult)
                  << std::endl;
        exit(EXIT_FAILURE);
    }

    cudaErrResult = cudaMemcpy(devB, hostB, vecSize, cudaMemcpyHostToDevice);
    if (cudaErrResult != cudaSuccess)
    {
        std::cout << "FAILURE COPYING hostB TO devB!"
                  << std::endl
                  << "Error: " << cudaGetErrorString(cudaErrResult)
                  << std::endl;
        exit(EXIT_FAILURE);
    }

    // compute how many blocks and threads we'll need to execute this kernel;
    // kernels utilize blocks of threads, where each block can be arranged into
    // a grid of blocks
    int threadsPerBlock = 256; // NOTE this is conventional - CLI param for kicks?
    int blocksPerGrid = (nElements + threadsPerBlock - 1) / threadsPerBlock;

    // call computation - performed on GPU
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    printf("Kernel launched with vanilla addition\n");

    // start time
    nSecondsStart = get_nanos();

    // launch on device
    vecAddKernel<<<blocksPerGrid, threadsPerBlock>>>(devA, devB, devC);

    // end time
    nSecondsEnd = get_nanos();

    printf("Finished in ... %ld nanoseconds\n", (nSecondsEnd-nSecondsStart));
    cudaErrResult = cudaGetLastError();
    if (cudaErrResult != cudaSuccess)
    {
        std::cout << "FAILED TO LAUNCH KERNEL! "
                  << "(error code " << cudaGetErrorString(cudaErrResult) << ")"
                  << std::endl;
        exit(EXIT_FAILURE);
    }

    // copy the result on device memory back to host
    cudaErrResult = cudaMemcpy(hostC, devC, vecSize, cudaMemcpyDeviceToHost);
    if (cudaErrResult != cudaSuccess)
    {
        std::cout << "FAILED TO COPY VECTOR C FROM DEVICE TO HOST! "
                  << "(error code " << cudaGetErrorString(cudaErrResult) << ")"
                  << std::endl;
        exit(EXIT_FAILURE);
    }

    // Verify that the result vector is correct
    for (int i = 0; i < nElements; ++i)
    {
        if (fabs(hostA[i] + hostB[i] - hostC[i]) > 1e-5)
        {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }

    // call using __fadd_rn intrinsic
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    printf("Kernel launched with __fadd_rn instrinsic addition\n");
    nSecondsStart = get_nanos();
    vecAddIntrinsicKernel<<<blocksPerGrid, threadsPerBlock>>>(devA, devB, devC);
    nSecondsEnd = get_nanos();
    printf("Finished in ... %ld nanoseconds\n", (nSecondsEnd-nSecondsStart));
    cudaErrResult = cudaGetLastError();
    if (cudaErrResult != cudaSuccess)
    {
        std::cout << "FAILED TO LAUNCH KERNEL! "
                  << "(error code " << cudaGetErrorString(cudaErrResult) << ")"
                  << std::endl;
        exit(EXIT_FAILURE);
    }

    // copy the result on device memory back to host
    cudaErrResult = cudaMemcpy(hostC, devC, vecSize, cudaMemcpyDeviceToHost);
    if (cudaErrResult != cudaSuccess)
    {
        std::cout << "FAILED TO COPY VECTOR C FROM DEVICE TO HOST! "
                  << "(error code " << cudaGetErrorString(cudaErrResult) << ")"
                  << std::endl;
        exit(EXIT_FAILURE);
    }

    // Verify that the result vector is correct
    for (int i = 0; i < nElements; ++i)
    {
        if (fabs(hostA[i] + hostB[i] - hostC[i]) > 1e-5)
        {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }

    // free CUDA device memory
    cudaErrResult = cudaFree(devA);
    if (cudaErrResult != cudaSuccess)
    {
        std::cout << "CUDA ERROR: COULD NOT FREE DEVICE MEMORY!" << std::endl;
        exit(EXIT_FAILURE);
    }

    cudaErrResult = cudaFree(devB);
    if (cudaErrResult != cudaSuccess)
    {
        std::cout << "CUDA ERROR: COULD NOT FREE DEVICE MEMORY!" << std::endl;
        exit(EXIT_FAILURE);
    }

    cudaErrResult = cudaFree(devC);
    if (cudaErrResult != cudaSuccess)
    {
        std::cout << "CUDA ERROR: COULD NOT FREE DEVICE MEMORY!" << std::endl;
        exit(EXIT_FAILURE);
    }

    // free host memory
    free(hostA);
    free(hostB);
    free(hostC);

    return 0;
}
