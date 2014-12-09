///////////////////////////////////////////////////////////////////////////////////////////////////
// Implementation of convolution on GPU
// CudaUtilities.cu
// 
// 
// This file contains helper functions for kernel/wrapper implementations
//
//
///////////////////////////////////////////////////////////////////////////////////////////////////

#include "CudaUtilities.h"

Timer *profilingTimer;
Timer *profilingTimer2;


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        //if (abort) exit(code);
    }
}

//Timer methods
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


// GPU memory handling functions

// Alloc device memory and set to 0
void
AllocateGPUMemory(void** ptr, unsigned int total_size, bool clear)
{
   cudaMalloc(ptr, total_size);
   if(clear) cudaMemset(*ptr, 0, total_size);
}

// Copy H->D(HtoD = true) and D->H(HtoD = false)
void
CopyGPUMemory(void* dest, void* src, unsigned int num_elems, bool HtoD)
{
   if (HtoD) cudaMemcpy(dest, src, num_elems, cudaMemcpyHostToDevice);
   else  gpuErrchk(cudaMemcpy(dest, src, num_elems, cudaMemcpyDeviceToHost));
}


void
CopyToGPUConstantMemory(void* dest, void* src, int numBytes)
{
    cudaMemcpyToSymbol(dest, src, numBytes, 0, cudaMemcpyHostToDevice);
}


// Free
void
FreeGPUMemory(void* ptr)
{
   cudaFree(ptr);
}

// Verify results
bool 
VerifyComputedData(float* reference, float* data, int numElems)
{
   bool result = compareData(reference, data, numElems, 0.0001f, 0.0f);
   printf("VerifyComputedData: %s\n", (result) ? "PASSED" : "FAILED");
   return result;
}