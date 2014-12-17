///////////////////////////////////////////////////////////////////////////////////////////////////
// Implementation of convolution on GPU
// CudaConvolve.h
// 
// 
// This file contains prototypes and types used by CudaConvolve.cu
//
//
//
// Created: 3-Dec-2014
//
///////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef CUDA_CONVOLVE_H
#define CUDA_CONVOLVE_H


#include "Image.h"
#include "Error.h"
#include "Convert.h"
#include "CudaUtilities.h"


///////////////////////////////////////////////////////////////////////////////////////////////////
// Convolution kernels that will go to constant memory
///////////////////////////////////////////////////////////////////////////////////////////////////
#define KERNEL_14641_X 5
#define KERNEL_14641_Y 5
#define KERNEL_121_X 3
#define KERNEL_121_Y 3

#define BLOCK_SIZE 16
#define TILE_SIZE (BLOCK_SIZE-KERNEL_14641_X+1)

//DBG
/*#define TESTKERN 3
#define TESTBLOCKSIZE 5
#define TESTTILESIZE (TESTBLOCKSIZE-TESTKERN+1)

const float TestKern[3][3] = { 1, 2, 1, 1, 2, 1, 1, 2, 1 };
__constant__ float Test[3][3];*/
//DBG END

enum BinomialFilterType{
    BINOMIAL121 = 0,
    BINOMIAL6126,
    BINOMIAL14641
};



///////////////////////////////////////////////////////////////////////////////////////////////////
// Prototypes
///////////////////////////////////////////////////////////////////////////////////////////////////
void
CudaConvolve2DRow(CFloatImage& buffer, CFloatImage& kernel, float dst[], int n);

template <class T>
void CudaConvolveXY(CImageOf<T> src, CImageOf<T>& dst, BinomialFilterType filterType);


#endif
