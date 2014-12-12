///////////////////////////////////////////////////////////////////////////////////////////////////
// Implementation of convolution on GPU
// CudaConvolve.h
// 
// 
// This file contains prototypes and types used by CudaConvolve.cpp
//
//
///////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef CUDA_CONVOLVE_H
#define CUDA_CONVOLVE_H


#include "Image.h"
#include "Error.h"
#include "Convert.h"
#include "CudaUtilities.h"

// Convolution kernels that will go to constant memory
#define KERNEL_14641_X 5
#define KERNEL_14641_Y 1
#define KERNEL_121_X 3
#define KERNEL_121_Y 1
#define KERNEL_8PTI_X 3
#define KERNEL_8PTI_Y 1

#define TILE_SIZE 128
#define BLOCK_14641 (TILE_SIZE + KERNEL_14641_X - 1)
#define DEPTH 16


void
CudaConvolve2DRow(CFloatImage& buffer, CFloatImage& kernel, float dst[], int n);

void
CopyToGPUConstantMem(void *dest, void *src, int num_elems);

#endif