///////////////////////////////////////////////////////////////////////////////////////////////////
// Implementation of convolution on GPU
// CudaConvolve.cu
// 
// 
// This file contains implementations of CUDA convolution kernels and wrappers
//
//
//
//
// Created: 3-Dec-2014
//
///////////////////////////////////////////////////////////////////////////////////////////////////
#include "assert.h"
#include "CudaConvolve.h"

// Profiling timers
extern Timer* profilingTimer2;

///////////////////////////////////////////////////////////////////////////////////////////////////
// Constant memory kernel definitions
///////////////////////////////////////////////////////////////////////////////////////////////////
__constant__ float ConvKern14641XY[KERNEL_14641_X][KERNEL_14641_Y];
__constant__ float ConvKern6126XY[KERNEL_14641_X][KERNEL_121_Y];

// 5x5 4th order Binomial filter 1/16[1;4;6;4;1] x 1/16[1,4,6,4,1]
const float Binomial14641XY[KERNEL_14641_X][KERNEL_14641_Y] = { 0.00390625, 0.015625, 0.0234375, 0.015625, 0.00390625,
                                                                0.015625,	0.0625,	0.09375,	0.0625,	0.015625,
                                                                0.0234375,	0.09375,	0.140625,	0.09375,	0.0234375,
                                                                0.015625,	0.0625,	0.09375,	0.0625,	0.015625,
                                                                0.00390625,	0.015625,	0.0234375,	0.015625,	0.00390625 };

// Mixed binomial 2nd and 4th order 5x3 filter 1/16[1;4;6;4;1] x 1/4[1,2,1]
const float Binomial6126XY[KERNEL_14641_X][KERNEL_121_Y] = { 0.015625,	0.03125,	0.015625,
                                                             0.0625,	0.125,	0.0625,
                                                             0.09375,	0.1875,	0.09375,
                                                             0.0625,	0.125,	0.0625,
                                                             0.015625,	0.03125,	0.015625 };


///////////////////////////////////////////////////////////////////////////////////////////////////
// Kernel Prototypes
///////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T>
__global__ void ConvolveXY(T* DevInBuffer, T* DevOutBuffer, CShape srcShape);

template <typename T>
__global__ void ConvolveXY121(T* DevInBuffer, T* DevOutBuffer, CShape srcShape);


///////////////////////////////////////////////////////////////////////////////////////////////////
// Kernel definitions
///////////////////////////////////////////////////////////////////////////////////////////////////

// 2D XY convolution for 4th order binomial filtering
template <typename T>
__global__ void ConvolveXY(T* DevInBuffer, T* DevOutBuffer, CShape srcShape)
{
    // Allow input tile to be stored in shared memory - Input is bigger than output tile
    __shared__ T Img[BLOCK_SIZE][BLOCK_SIZE];

    // Short notations
    int tix = threadIdx.x, tiy = threadIdx.y;
    float Sum;

    // Offset is basically the dimension of the extending halo
    int Offset = (int)KERNEL_14641_X / 2;

    int N_x = blockIdx.x * TILE_SIZE + tix - Offset;
    int N_y = blockIdx.y * TILE_SIZE + tiy - Offset;

    // Compute absolute indices just once for each thread
    int Abs_x = N_x * srcShape.nBands;
    int Abs_y = N_y * srcShape.nBands * srcShape.width;

    // Compute right bound
    int RBound = BLOCK_SIZE - Offset;

    // Subtract just once for each thread
    int OffX = tiy - Offset;
    int OffY = tix - Offset;

    // Go over all depth layers (channels)
    for (int c = 0; c < srcShape.nBands; c++)
    {
       // If in bound of input N, then load the element, else load zero
       if ((N_x >= 0 && N_x < srcShape.width) && (N_y >= 0 && N_y < srcShape.height))
       {
           Img[tiy][tix] = DevInBuffer[Abs_y + Abs_x + c];
       }
       else
       {
           Img[tiy][tix] = 0.0f;
       }
       
       // Wait for all threads to load
       __syncthreads();

       // Clear sum
       Sum = 0.0f;

       // Multiply and accumulate - filtering non-halo threads in each block
       if ((tix >= Offset && tix < RBound) && (tiy >= Offset && tiy < RBound))
       {
           for (int i = 0; i < KERNEL_14641_X; i++)
           {
               for (int j = 0; j < KERNEL_14641_Y; j++)
               {
                   Sum += ConvKern14641XY[i][j] * Img[OffX + i][OffY + j];
               }
           }
       
           // Store the sum in P - Subset because N_x and N_y dest indices are applicable only to threads performing computation
           if ((N_x >= 0 && N_x < srcShape.width) && (N_y >= 0 && (N_y < srcShape.height)))
           {
               DevOutBuffer[Abs_y + Abs_x + c] = Sum;
           }
       }

       // Wait for all threads to finish computing
       __syncthreads();
    }
}


// 2D XY convolution for mixed 2nd and 4th order binomial filterings
template <typename T>
__global__ void ConvolveXY121(T* DevInBuffer, T* DevOutBuffer, CShape srcShape)
{
    // Allow input tile to be stored in shared memory - Input is bigger than output tile
    __shared__ T Img[BLOCK_SIZE][BLOCK_SIZE];

    // Short notations
    int tix = threadIdx.x, tiy = threadIdx.y;
    float Sum;

    // Offset is basically the dimension of the extending halo
    int OffsetX = (int)KERNEL_121_X / 2;
    int OffsetY = (int)KERNEL_14641_Y / 2;

    int N_x = blockIdx.x * TILE_SIZE + tix - OffsetX;
    int N_y = blockIdx.y * TILE_SIZE + tiy - OffsetY;

    // Compute absolute indices just once for each thread
    int Abs_x = N_x * srcShape.nBands;
    int Abs_y = N_y * srcShape.nBands * srcShape.width;

    // Compute right bound and lower(down) bound
    int RBound = BLOCK_SIZE - OffsetX;
    int DBound = BLOCK_SIZE - OffsetY;

    // Subtract just once for each thread
    int OffX = tiy - OffsetY;
    int OffY = tix - OffsetX;

    // Go over all depth layers (channels)
    for (int c = 0; c < srcShape.nBands; c++)
    {
        // If in bound of input N, then load the element, else load zero
        if ((N_x >= 0 && N_x < srcShape.width) && (N_y >= 0 && N_y < srcShape.height))
        {
            Img[tiy][tix] = DevInBuffer[Abs_y + Abs_x + c];
        }
        else
        {
            Img[tiy][tix] = 0.0f;
        }

        // Wait for all threads to load
        __syncthreads();

        // Clear sum
        Sum = 0.0f;

        // Multiply and accumulate - filtering non-halo threads in each block
        if ((tix >= OffsetX && tix < RBound) && (tiy >= OffsetY && tiy < DBound))
        {
            for (int i = 0; i < KERNEL_14641_X; i++)
            {
                for (int j = 0; j < KERNEL_121_Y; j++)
                {
                    Sum += ConvKern6126XY[i][j] * Img[OffX + i][OffY + j];
                }
            }

            // Store the sum in P - Subset because N_x and N_y dest indices are applicable only to threads performing computation
            if ((N_x >= 0 && N_x < srcShape.width) && (N_y >= 0 && (N_y < srcShape.height)))
            {
                DevOutBuffer[Abs_y + Abs_x + c] = Sum;
            }
        }

        // Wait for all threads to finish computing
        __syncthreads();
    }
}



///////////////////////////////////////////////////////////////////////////////////////////////////
// CPU-GPU Host Wrappers
///////////////////////////////////////////////////////////////////////////////////////////////////
template <class T>
void CudaConvolveXY(CImageOf<T> src, CImageOf<T>& dst, BinomialFilterType filterType)
{
    // Extract kernel details
    CShape srcShape = src.Shape();

    // GPU memories
    T *DevInBuffer, *DevOutBuffer;

    //DBG
    //srcShape.width = 13;
    //srcShape.height = 14;
    
    profilingTimer2->startTimer();

    // Allocate memory to copy all image/cost-map channels to GPU
    int AllocSize = sizeof(T) * srcShape.width * srcShape.height * srcShape.nBands;
    AllocateGPUMemory((void**)&DevInBuffer, AllocSize, false);

    // Allocate memory for output
    AllocateGPUMemory((void**)&DevOutBuffer, AllocSize, false);

    // Transfer everything to GPU
    T *StartAddr = &src.Pixel(0, 0, 0);
    CopyGPUMemory((void*)DevInBuffer, (void*)StartAddr, AllocSize, true);
    printf("\nMemCpy to GPU time = %f ms\n", profilingTimer2->stopAndGetTimerValue());

    // Set kernel parameters and launch kernel
    dim3 Grid(ceil((float)srcShape.width / TILE_SIZE), ceil((float)srcShape.height / TILE_SIZE), 1);
    dim3 Block(BLOCK_SIZE, BLOCK_SIZE, 1);

    // Copy the kernel to constant memory and execute kernel
    // GPUERRORCHECK(cudaMemcpyToSymbol(Test, TestKern, sizeof(float) * TESTKERN * TESTKERN, 0, cudaMemcpyHostToDevice))
    profilingTimer2->startTimer();
    switch (filterType)
    {
        case BINOMIAL6126:
        {
           GPUERRORCHECK(cudaMemcpyToSymbol(ConvKern6126XY, Binomial6126XY, sizeof(float) * KERNEL_121_X * KERNEL_14641_Y, 0, cudaMemcpyHostToDevice))
           ConvolveXY121 << <Grid, Block >> >(DevInBuffer, DevOutBuffer, srcShape);
           break;
        }
        case BINOMIAL14641:
        {
           GPUERRORCHECK(cudaMemcpyToSymbol(ConvKern14641XY, Binomial14641XY, sizeof(float) * KERNEL_14641_X * KERNEL_14641_Y, 0, cudaMemcpyHostToDevice))
           ConvolveXY << <Grid, Block >> >(DevInBuffer, DevOutBuffer, srcShape);
           break;
        }
        default:
        {  
           // Unsupported kernel case - this path should never be hit
           throw CError("Convolution kernel type unknown");
        }
    }
    
    // Wait for all blocks to finish
    cudaDeviceSynchronize();
    printf("\nConvolve kernel execution time = %f ms\n", profilingTimer2->stopAndGetTimerValue());

    // Copy computed elements back to CPU memory
    profilingTimer2->startTimer();
    dst.ReAllocate(srcShape, false);
    T *DestStartAddr = &dst.Pixel(0, 0, 0);
    CopyGPUMemory((void*)DestStartAddr, (void*)DevOutBuffer, AllocSize, false);
    printf("\nMemCpy from GPU time = %f ms\n", profilingTimer2->stopAndGetTimerValue());

    // Free GPU memory
    FreeGPUMemory(DevInBuffer);
    FreeGPUMemory(DevOutBuffer);
}




///////////////////////////////////////////////////////////////////////////////////////////////////
// Templated functions' instantiation
///////////////////////////////////////////////////////////////////////////////////////////////////
template void CudaConvolveXY<float>(class CImageOf<float>, class CImageOf<float> &, BinomialFilterType);
template void CudaConvolveXY<unsigned char>(class CImageOf<unsigned char>, class CImageOf<unsigned char> &, BinomialFilterType);
template void CudaConvolveXY<int>(class CImageOf<int>, class CImageOf<int> &, BinomialFilterType);

template __global__ void ConvolveXY<float>(float *, float *, struct CShape);
