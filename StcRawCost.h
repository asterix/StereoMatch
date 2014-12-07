#include <cuda.h>
#if defined(WIN32) ||  CUDAVER >= 5
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#endif

#include "StereoParameters.h"

#ifndef STCRAWCOST
#define STCRAWCOST

#define BLOCK_SIZE 64

__global__ void InterpolateLineKernel(int* buf, int s, int w, int nB, EStereoInterpFn match_interp)

#endif
