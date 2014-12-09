///////////////////////////////////////////////////////////////////////////////////////////////////
// Implementation of convolution on GPU
// CudaUtilities.h
// 
// 
// This file contains helper functions for kernel/wrapper implementations
//
//
// Created: 3-Dec-2014
//
//
///////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CUDAUTILITIES_H
#define CUDAUTILITIES_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <helper_cuda.h>
#include <helper_functions.h>
#include <helper_timer.h>
#include <driver_types.h>


#define GPUERRORCHECK(call) { ReportGPUError((call), __FILE__, __LINE__);}
inline void ReportGPUError(cudaError_t ret_code, const char *file, int line)
{
    if (ret_code != cudaSuccess)
    {
        fprintf(stderr, "ReportGPUError: %s %s %d\n", cudaGetErrorString(ret_code), file, line);
        exit(1);
    }
}


// Timer utility
class Timer
{
private:
    unsigned int Timer1;
    StopWatchInterface *timerIfc;

public:

    Timer()
    {
        sdkCreateTimer(&timerIfc);
        sdkResetTimer(&timerIfc);
    }

    ~Timer()
    {
        //Nothing here
    }

    void startTimer();
    float stopAndGetTimerValue();
};

// CUDA Memory handling functions
void
CopyGPUMemory(void* dest, void* src, unsigned int num_elems, bool HtoD);

void
AllocateGPUMemory(void** ptr, unsigned int total_size, bool clearMemory);

void
FreeGPUMemory(void* ptr);

void
CopyToGPUConstantMemory(void* dest, void* src, int numBytes);

// Computed data verification
bool
VerifyComputedData(float* reference, float* data, int numElems);

#endif