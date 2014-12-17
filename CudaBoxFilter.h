///////////////////////////////////////////////////////////////////////////////////////////////////
// Implementation of convolution on GPU
// CudaBoxFilter.h
// 
// 
// This file contains prototypes and types used by CudaBoxFilter.cu
//
//
//
// Created: 13-Dec-2014
//
///////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef CUDA_BOXFILTER_H
#define CUDA_BOXFILTER_H


#include "Image.h"
#include "Error.h"
#include "Convert.h"
#include "CudaUtilities.h"

// Define Box window size
// Note: Trying make this a dynamic variable results in performance decrease since shared memory array has to be 1D
#define BOX_WINDOW_SIZE 5

// Box filtering window sizes and CUDA tile sizes
// We define BLOCK_SIZE and not TILE_SIZE since BLOCK_SIZE fixes the threads per block
#define BOX_BLOCK_SIZE 16
#define BOX_TILE_SIZE (BOX_BLOCK_SIZE-BOX_WINDOW_SIZE+1)

template <class T>
void CudaBoxFilterXY(CImageOf<T> src, CImageOf<T>& dst, int WindowSize);


#endif