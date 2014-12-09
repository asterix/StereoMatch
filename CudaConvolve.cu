///////////////////////////////////////////////////////////////////////////////////////////////////
// Implementation of convolution on GPU
// CudaConvolve.cu
// 
// 
// This file contains implementations of CUDA kernels and wrappers
//
//
///////////////////////////////////////////////////////////////////////////////////////////////////
#include "CudaConvolve.h"

// Constant memory kernel definitions
__constant__ float ConvKern14641[KERNEL_14641_X];
__constant__ float ConvKern121[KERNEL_121_X];
__constant__ float ConvKern8PTI[KERNEL_8PTI_X];


//Prototypes
__global__ void ConvolveOneX(float* DevInBuffer, float* DevOutBuffer, int numElems, CShape kernel, CShape buffer);


// 2D Row convolution
__global__ void ConvolveOneX(float* DevInBuffer, float* DevOutBuffer, int numElems, CShape kernel, CShape buffer)
{
    __shared__ float InBuffer[BLOCK_14641];

    int tix = threadIdx.x;
    int numChans = buffer.nBands;
    int absIdx = blockIdx.x * TILE_SIZE + tix;
    float sum;
    int Offset = KERNEL_14641_X / 2;
    int loadIdx = absIdx - Offset;


    for (int i = 0; i < numChans; i++)
    {
       // Load input elements to shared memory
       if (loadIdx >= 0 && loadIdx < buffer.width)
       {
           InBuffer[tix] = DevInBuffer[loadIdx*numChans+i];
       }
       else
       {
           // These are dummy out-of-bound elements. Ghost elements are already filled.
           InBuffer[tix] = 0;
       }
       
       // Wait for all threads to finish loading
       __syncthreads();

       sum = 0;
       // Convolve
       if((tix >= Offset) && (tix < BLOCK_14641 - Offset))
       {
           for (int k = 0; k < kernel.width; k++)
           {
               sum += ConvKern14641[k] * InBuffer[tix - Offset + k];
           }

           if (loadIdx >= 0 && loadIdx < buffer.width)
           {
               // For the first and the last block don't store the dummy element indices
               DevOutBuffer[loadIdx*numChans + i] = sum;
           }
       }

       // Wait for all thread to compute before loading the next channel
       __syncthreads();

    }
}

void
CudaConvolve2DRow(CFloatImage buffer, CFloatImage kernel, float *dest, int numElems)
{
   // Extract kernel details
   CShape kerShape = kernel.Shape();
   CShape srcShape = buffer.Shape();
   int kernelWidth = kerShape.width;
   int kernelHeight = kerShape.height;

   // Extra input buffer details
   int numChans = buffer.Shape().nBands;
   int numTotElems = buffer.Shape().width;

   // One thread per pixel - processes all channels
   dim3 Grid(ceil((float)numTotElems / TILE_SIZE), 1, 1);
   dim3 Block(BLOCK_14641, 1, 1);

   // Get the row start address
   float *StartAddr = &buffer.Pixel(0, 0, 0);
   float *KernStartAddr = &kernel.Pixel(0, 0, 0);
   float *DevInBuffer, *DevOutBuffer;

   // Each pixel has numChans channels
   AllocateGPUMemory((void**)&DevInBuffer, sizeof(float) * numTotElems * numChans * kernelHeight, false);
   AllocateGPUMemory((void**)&DevOutBuffer, sizeof(float) * (numTotElems * numChans * kernelHeight + KERNEL_14641_X), false);
   CopyGPUMemory(DevInBuffer, StartAddr, sizeof(float) * numTotElems * numChans * kernelHeight, true);


   cudaMemcpyToSymbol(ConvKern14641, KernStartAddr, sizeof(float) * kernelWidth, 0, cudaMemcpyHostToDevice);
   //CopyToGPUConstantMem(ConvKern14641, KernStartAddr, sizeof(float) * kernelWidth);

   ConvolveOneX << <Grid, Block >> >(DevInBuffer, DevOutBuffer, numElems, kerShape, srcShape);

   cudaDeviceSynchronize();
   CopyGPUMemory((void*)dest, (void*)DevOutBuffer, sizeof(float) * (numTotElems * numChans * kernelHeight + KERNEL_14641_X), false);
}

void
CopyToGPUConstantMem(void *dest, void *src, int num_elems)
{
   cudaMemcpyToSymbol(dest, src, num_elems, 0, cudaMemcpyHostToDevice);
}