///////////////////////////////////////////////////////////////////////////////////////////////////
// Implementation of convolution on GPU
// CudaBoxFilter.cu
// 
// 
// This file contains implementations of CUDA boxfiltering kernels and wrappers
//
//
//
//
// Created: 13-Dec-2014
//
///////////////////////////////////////////////////////////////////////////////////////////////////
#include "CudaBoxFilter.h"

// Profiling timers
extern Timer* profilingTimer2;
extern bool ZeroCopySupported;

                                                               
///////////////////////////////////////////////////////////////////////////////////////////////////
// Kernel Prototypes
///////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T>
__global__ void BoxFilterXY(T* DevInBuffer, T* DevOutBuffer, CShape srcShape);



///////////////////////////////////////////////////////////////////////////////////////////////////
// Kernel definitions
///////////////////////////////////////////////////////////////////////////////////////////////////

// BoxFiltering over variable window sizes
template <typename T>
__global__ void BoxFilterXY(T* DevInBuffer, T* DevOutBuffer, CShape srcShape)
{
    // Allow input tile to be stored in shared memory - Input is bigger than output tile
    __shared__ T Img[BOX_BLOCK_SIZE][BOX_BLOCK_SIZE];

    // Short notations
    int tix = threadIdx.x, tiy = threadIdx.y;
    float Sum;

    // Offset is basically the dimension of the extending halo
    int Offset = (int)BOX_WINDOW_SIZE / 2;

    int N_x = blockIdx.x * BOX_TILE_SIZE + tix - Offset;
    int N_y = blockIdx.y * BOX_TILE_SIZE + tiy - Offset;

    // Compute absolute indices just once for each thread
    int Abs_x = N_x * srcShape.nBands;
    int Abs_y = N_y * srcShape.nBands * srcShape.width;

    // Compute right bound
    int RBound = BOX_BLOCK_SIZE - Offset;

    // Subtract just once for each thread
    int OffX = tiy - Offset;
    int OffY = tix - Offset;

    // Boxfilter averaging elements
    int Avg = BOX_WINDOW_SIZE * BOX_WINDOW_SIZE;

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

        // Accumulate and average - filtering non-halo threads in each block
        if ((tix >= Offset && tix < RBound) && (tiy >= Offset && tiy < RBound))
        {
            for (int i = 0; i < BOX_WINDOW_SIZE; i++)
            {
                for (int j = 0; j < BOX_WINDOW_SIZE; j++)
                {
                    Sum += Img[OffX + i][OffY + j];
                }
            }

            // Store the sum in P - Subset because N_x and N_y dest indices are applicable only to threads performing computation
            if ((N_x >= 0 && N_x < srcShape.width) && (N_y >= 0 && (N_y < srcShape.height)))
            {
                DevOutBuffer[Abs_y + Abs_x + c] = Sum/Avg;
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
void CudaBoxFilterXY(CImageOf<T> src, CImageOf<T>& dst, int WindowSize)
{
    // Extract kernel details
    CShape srcShape = src.Shape();

    // GPU memories
    T *DevInBuffer, *DevOutBuffer;

    profilingTimer2->startTimer();
    // Get source and destination pointers
    int AllocSize;
    T *StartAddr = &src.Pixel(0, 0, 0);

    dst.ReAllocate(srcShape, false);
    T *DestStartAddr = &dst.Pixel(0, 0, 0);

    // Allocate memory to copy all image/cost-map channels to GPU
    if (!ZeroCopySupported)
    {
       AllocSize = sizeof(T) * srcShape.width * srcShape.height * srcShape.nBands;
       // Allocate in-data memory
       AllocateGPUMemory((void**)&DevInBuffer, AllocSize, false);
       // Allocate memory for output
       AllocateGPUMemory((void**)&DevOutBuffer, AllocSize, false);

       // Transfer everything to GPU
       CopyGPUMemory((void*)DevInBuffer, (void*)StartAddr, AllocSize, true);
    }
    else
    {
        // Zero-Copy - No Alloc - No Copy - No Free
        // Fast direct page-locked CPU-GPU access in unified physical memory architecture
        GPUERRORCHECK(cudaHostGetDevicePointer((void **)&DevInBuffer, (void *)StartAddr, 0))
        GPUERRORCHECK(cudaHostGetDevicePointer((void **)&DevOutBuffer, (void *)DestStartAddr, 0))
    }
    printf("\nMemCpy to GPU time = %f ms\n", profilingTimer2->stopAndGetTimerValue());

    // Set kernel parameters and launch kernel
    dim3 Grid(ceil((float)srcShape.width / BOX_TILE_SIZE), ceil((float)srcShape.height / BOX_TILE_SIZE), 1);
    dim3 Block(BOX_BLOCK_SIZE, BOX_BLOCK_SIZE, 1);

    profilingTimer2->startTimer();
    GPUERRORCHECK(cudaFuncSetCacheConfig(BoxFilterXY<float>, cudaFuncCachePreferL1))
    BoxFilterXY<< <Grid, Block >> >(DevInBuffer, DevOutBuffer, srcShape);

    // Wait for all blocks to finish
    cudaDeviceSynchronize();
    printf("\nBoxFilter kernel execution time = %f ms\n", profilingTimer2->stopAndGetTimerValue());

    // Copy computed elements back to CPU memory
    profilingTimer2->startTimer();
    // Only if traditional copy-back is needed
    if (!ZeroCopySupported)
    {
       CopyGPUMemory((void*)DestStartAddr, (void*)DevOutBuffer, AllocSize, false);
       
       // Free GPU memory
       FreeGPUMemory(DevInBuffer);
       FreeGPUMemory(DevOutBuffer);
    }
    printf("\nMemCpy from GPU time = %f ms\n", profilingTimer2->stopAndGetTimerValue());

}



///////////////////////////////////////////////////////////////////////////////////////////////////
// Templated functions' instantiation
///////////////////////////////////////////////////////////////////////////////////////////////////
template void CudaBoxFilterXY<float>(class CImageOf<float>, class CImageOf<float> &, int);
template void CudaBoxFilterXY<unsigned char>(class CImageOf<unsigned char>, class CImageOf<unsigned char> &, int);
template void CudaBoxFilterXY<int>(class CImageOf<int>, class CImageOf<int> &, int);

template __global__ void BoxFilterXY<float>(float *, float *, struct CShape);
