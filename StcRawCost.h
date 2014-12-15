
#ifndef STCRAWCOST_H
#define STCRAWCOST_H

#include "StereoParameters.h"
#include "CudaUtilities.h"

#define BLOCKSIZE 512
#define BAD_COST -1

#define UNDEFINED_COST true // set this to true to pad with outside_cost

struct ImageSizeStruct
{
    int bandSize;
    int rowSize;
    int pixSize;
    int width;
    int height;
    int bands;
};

struct ImageStructUChar
{
    ImageSizeStruct imageSize;
    uchar* image;
};

struct ImageStructFloat
{
    ImageSizeStruct imageSize;
    float* image;
};

struct TwoDIntArray
{
    int* array;
    int width;
    int height;
    int num_elems;
    int size_bytes; // in bytes
};

struct TwoDFloatArray
{
    float* array;
    int width;
    int height;
    int num_elems;
    int size_bytes; // in bytes
};

struct MatchLineStruct
{
    int w;
    int b;
    int interpolated;
    int* rmn; // min/max of ref (ref if rmx == 0)
    int* rmx;
    int* mmn; // min/max of mtc (mtc if mmx == 0)
    int* mmx;
    int m_disp_n;
    int disp;
    int m_disp_den;
    EStereoMatchFn match_fn; // matching function
    int match_max; // maximum difference for truncated SAD/SSD
    float match_outside;
};

struct LineProcessStruct
{
    int m_disp_den;
    int m_disp_n;
    int b;
    int w;
    int h;
    EStereoInterpFn match_interp;
    int match_interval;
    int match_interpolated;
    int m_frame_diff_sign;
    int disp_min;
    int m_disp_num;
    EStereoMatchFn match_fn;
    int match_max;
    float match_outside;
    int n_interp;
};

struct BufferStruct
{
    TwoDIntArray buffer0;
    TwoDIntArray buffer1;
    TwoDIntArray min_bf0;
    TwoDIntArray max_bf0;
    TwoDIntArray min_bf1;
    TwoDIntArray max_bf1;

    TwoDFloatArray cost1;
};

// Raw Cost standard functions
__host__ __device__ void InterpolateLine(int buf[], int s, int w, int nB, EStereoInterpFn match_interp);
__host__ __device__ void BirchfieldTomasiMinMax(const int* buffer, int* min_buf, int* max_buf, const int w, const int b);
__device__ void MatchLineDevice(MatchLineStruct args, float* cost, float* cost1_in);
void MatchLineHost(MatchLineStruct args, float* cost);

// Raw Cost helper functions
__host__ __device__ float CubicInterpolateRC(float x0, float v0, float v1, float v2, float v3);
__device__ int PixelCoordToAbs(ImageSizeStruct size, int x, int y, int band);
__device__ uchar* PixelAddress(ImageStructUChar image, int x, int y, int band);
__device__ float* PixelAddress(ImageStructFloat image, int x, int y, int band);
ImageSizeStruct PopulateImageSizeStruct(CImage image);
void Populate2DArray(TwoDIntArray* value, int width, int height);
void Populate2DArray(TwoDFloatArray* value, int width, int height);

// Raw Cost kernel functions
void LineProcess(CByteImage m_reference, CByteImage m_matching, CFloatImage m_cost, LineProcessStruct args);
__global__ void LineProcessKernel(ImageStructUChar m_reference, ImageStructUChar m_matching, ImageStructFloat m_cost,
    BufferStruct buffs, LineProcessStruct args);



#endif
