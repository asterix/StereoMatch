///////////////////////////////////////////////////////////////////////////////////////////////////
// Implementation of convolution on GPU
// CudaUtilities.cu
// 
// 
// This file contains helper functions for kernel/wrapper implementations
//
//
// Use GPUERRORCHECK to verify successful completion of CUDA calls
//
// Created: 3-Dec-2014
//
///////////////////////////////////////////////////////////////////////////////////////////////////

#include "CudaUtilities.h"

Timer *profilingTimer;
Timer *profilingTimer2;


///////////////////////////////////////////////////////////////////////////////////////////////////
//Timer methods
///////////////////////////////////////////////////////////////////////////////////////////////////
void 
Timer::startTimer()
{
   sdkStartTimer(&timerIfc);
}

float 
Timer::stopAndGetTimerValue()
{
   sdkStopTimer(&timerIfc);
   float time = sdkGetTimerValue(&timerIfc);
   sdkResetTimer(&timerIfc);
   return time;
}



///////////////////////////////////////////////////////////////////////////////////////////////////
// GPU memory handling functions
///////////////////////////////////////////////////////////////////////////////////////////////////

// Alloc device memory and set to 0
void
AllocateGPUMemory(void** ptr, unsigned int total_size, bool clear)
{
    GPUERRORCHECK(cudaMalloc(ptr, total_size))
    if (clear) GPUERRORCHECK(cudaMemset(*ptr, 0, total_size))
}

// Copy to device and back
// H->D(HtoD = true) and D->H(HtoD = false)
void
CopyGPUMemory(void* dest, void* src, unsigned int num_elems, bool HtoD)
{
    if (HtoD) GPUERRORCHECK(cudaMemcpy(dest, src, num_elems, cudaMemcpyHostToDevice))
   else  GPUERRORCHECK(cudaMemcpy(dest, src, num_elems, cudaMemcpyDeviceToHost))
}

// Copy to constant memory
void
CopyToGPUConstantMemory(void* dest, void* src, int numBytes)
{
    GPUERRORCHECK(cudaMemcpyToSymbol(dest, src, numBytes, 0, cudaMemcpyHostToDevice))
}

// Free
void
FreeGPUMemory(void* ptr)
{
    GPUERRORCHECK(cudaFree(ptr))
}



///////////////////////////////////////////////////////////////////////////////////////////////////
// Verify results
///////////////////////////////////////////////////////////////////////////////////////////////////
bool 
VerifyComputedData(float* reference, float* data, int numElems)
{
   bool result = compareData(reference, data, numElems, 0.0001f, 0.0f);
   printf("VerifyComputedData: %s\n", (result) ? "DATA OK" : "DATA MISMATCH");
   return result;
}
