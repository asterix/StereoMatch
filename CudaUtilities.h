///////////////////////////////////////////////////////////////////////////////////////////////////
// Implementation of convolution on GPU
// CudaUtilities.h
// 
// 
// This file contains helper functions for kernel/wrapper implementations
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


void
CopyGPUMemory(void* dest, void* src, unsigned int num_elems, bool HtoD);

void
AllocateGPUMemory(void** ptr, unsigned int total_size, bool clearMemory);

void
FreeGPUMemory(void* ptr);

void
CopyToGPUConstantMemory(void* dest, void* src, int numBytes);

bool
VerifyComputedData(float* reference, float* data, int numElems);

#endif