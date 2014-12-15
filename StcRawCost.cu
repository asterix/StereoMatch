// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "StereoMatcher.h"
#include "Warp1D.h"

#include <cuda.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "StcRawCost.h"

#define PREALLOC_BUFF (0) // size of pre-allocated buffers to use in each thread (set to 0 to use dynamic allocation)

// Serial / Parallel implementation

__host__ __device__ void InterpolateLine(int buf[], int s, int w, int nB, EStereoInterpFn match_interp)     // interpolation function
{
    // Interpolate the missing values
    float si = 1.0f / s;
    for (int x = 0; x < w - 1; x++)
    {
        for (int b = 0; b < nB; b++)
        {
            int* v = &buf[s*x*nB + b];
            float I0 = (float)v[0];
            float I1 = (float)v[s*nB];
            if (match_interp == eCubic) // cubic interpolation
            {
                float Im = (x > 0) ? v[-s*nB] :
                    (I0 - (I1 - I0));  // extend linearly
                float Ip = (x + 1 < w - 1) ? v[2 * s*nB] :
                    (I1 + (I1 - I0));  // extend linearly
                float sf = si;
                for (int is = 1; is < s; is++, sf += si)
                {
                    v += nB;
                    float Ii = CubicInterpolateRC(sf, Im, I0, I1, Ip);
                    v[0] = int(Ii);
                }
            }
            else  // linear interpolation
            {
                float d = (I1 - I0) / (float)s;
                for (int is = 1; is < s; is++)
                {
                    v += nB;
                    I0 += d;
                    v[0] = int(I0);
                }
            }
        }
    }
}

__host__ __device__ float CubicInterpolateRC(float x0, float v0, float v1, float v2, float v3)
{
    // See Szeliski & Ito, IEE Proc 133(6) 1986.
    float x1 = 1.0f - x0;
    float s0 = v2 - v0;     // slope matches central difference
    float s1 = v1 - v3;     // slope matches central difference
    float d1 = v2 - v1;
    float phi0 = d1 * (x0 * x0) * (2.0f * x1 + 1.0f);
    float phi1a = s0 * x0 * (x1 * x1);
    float phi1b = s1 * x1 * (x0 * x0);
    float v = v1 + phi0 + phi1a + phi1b;
    return v;
}

__host__ __device__ void BirchfieldTomasiMinMax(const int* buffer, int* min_buf, int* max_buf, const int w, const int b)
{
    // Compute for every (interpolated) pixel, the minimum and maximum
    //  values in the two half-intervals before and after it
    //  (see [Birchfield & Tomasi, PAMI 20(40), April 1998, p. 401]).

    // Process each band separaterly
    for (int k = 0; k < b; k++)
    {
        int Ir = buffer[k], b1 = buffer[k];
        for (int x = 0, l = k; x < w; x++, l += b)
        {
            int Il = Ir, b0 = b1;   // shift down previously computed values
            if (x < w - 1)
                b1 = buffer[l + b];
            Ir = (b0 + b1 + 1) / 2;   // interpolated half-value
            min_buf[l] = __min(Il, __min(b0, Ir));
            max_buf[l] = __max(Il, __max(b0, Ir));
        }
    }
}

__device__ void MatchLineDevice(MatchLineStruct args, float* cost, float* cost1_in)
{
    // Set up the starting addresses, pointers, and cutoff value
    int n = (args.w - 1)*args.m_disp_den + 1;             // number of reference pixels
    int s = (args.interpolated) ? 1 : args.m_disp_den;     // skip in reference pixels
    int cutoff = (args.match_fn == eSD) ? args.match_max * args.match_max : abs(args.match_max);
    // TODO:  cutoff is not adjusted for the number of bands...

#if PREALLOC_BUFF
    float cost1[PREALLOC_BUFF];
#else
    float* cost1 = cost1_in;
#endif

    // Match valid pixels
    float  left_cost = BAD_COST;
    float right_cost = BAD_COST;

    for (int x = 0; x < n; x += s)
    {
        // Compute ref and match pointers
        cost1[x] = BAD_COST;
        int x_r = x, x_m = x + args.disp;
        if (x_m < 0 || x_m >= n)
            continue;
        int* rn = &(args.rmn[x_r*args.b]);    // pointer to ref or min pixel(s)
        int* rx = &(args.rmx[x_r*args.b]);    // pointer to ref    max pixel(s)
        int* mn = &(args.mmn[x_m*args.b]);    // pointer to mtc or min pixel(s)
        int* mx = &(args.mmx[x_m*args.b]);    // pointer to mtc    max pixel(s)
        int  diff_sum = 0;        // accumulated error

        // This code could be special-cased for b==1 for more efficiency...
        for (int ib = 0; ib < args.b; ib++)
        {
            int diff1 = mn[ib] - rn[ib];    // straightforward difference
            if (args.rmx && args.mmx)
            {
                // Compare intervals (see partial shuffle code in StcEvaluate.cpp)
                int xn = __max(rn[ib], mn[ib]);     // max of mins
                int nx = __min(rx[ib], mx[ib]);     // min of maxs
                if (xn <= nx)
                    diff1 = 0;          // overlapping ranges -> no error
                else
                    diff1 = (mn[ib] > rx[ib]) ?     // check sign
                    mn[ib] - rx[ib] :
                    rn[ib] - mx[ib];          // gap between intervals
            }
            int diff2 = (args.match_fn == eSD) ?    // squared or absolute difference
                diff1 * diff1 : abs(diff1);
            diff_sum += diff2;
        }
        int diff3 = __min(diff_sum, cutoff);    // truncated difference
        if (left_cost == BAD_COST)
            left_cost = diff3;  // first cost computed
        right_cost = diff3;     // last  cost computed
        cost1[x] = diff3;        // store in temporary array
    }

    // Fill in the left and right edges
    if (UNDEFINED_COST)
        left_cost = right_cost = args.match_outside;
    for (int x = 0; x < n && cost1[x] == BAD_COST; x += s)
        cost1[x] = left_cost;
    for (int x = n - 1; x >= 0 && cost1[x] == BAD_COST; x -= s)
        cost1[x] = right_cost;

    // Box filter if interpolated costs
    int dh = args.m_disp_den / 2;
    float box_scale = 1.0 / (2 * dh + 1);
    for (int x = 0, y = 0; y < args.w*args.m_disp_n; x += args.m_disp_den, y += args.m_disp_n)
    {
        if (args.interpolated && args.m_disp_den > 1)
        {
            float sum = 0;
            for (int k = -dh; k <= dh; k++)
            {
                int l = __max(0, __min(n - 1, x + k));  // TODO: make more efficient
                sum += cost1[l];
            }
            cost[y] = int(box_scale * sum + 0.5);
        }
        else
            cost[y] = cost1[x];
    }
}

// Kernel functions

__global__ void LineProcessKernel(ImageStructUChar m_reference, ImageStructUChar m_matching, ImageStructFloat m_cost,
    float* cost1, int cost1_width, BufferStruct buffs, LineProcessStruct args)
{
    unsigned y = (threadIdx.y + blockIdx.y * blockDim.y);

    // Process all of the lines
    if (y < args.h)
    {
        
        uchar* ref = PixelAddress(m_reference, 0, y, 0);
        uchar* mtc = PixelAddress(m_matching, 0, y, 0);

#if PREALLOC_BUFF
        int  buf0[PREALLOC_BUFF];
        int  buf1[PREALLOC_BUFF];
        int  min0[PREALLOC_BUFF];
        int  max0[PREALLOC_BUFF];
        int  min1[PREALLOC_BUFF];
        int  max1[PREALLOC_BUFF];
#else
        unsigned buf_start = y * args.n_interp * args.b;
        int*  buf0 = &(buffs.buffer0[buf_start]);
        int*  buf1 = &(buffs.buffer1[buf_start]);
        int*  min0 = &(buffs.min_bf0[buf_start]);
        int*  max0 = &(buffs.max_bf0[buf_start]);
        int*  min1 = &(buffs.min_bf1[buf_start]);
        int*  max1 = &(buffs.max_bf1[buf_start]);
#endif

        // Fill the line buffers
        for (int x = 0, l = 0, m = 0; x < args.w; x++, m += args.m_disp_den*args.b)
        {
            for (int k = 0; k < args.b; k++, l++)
            {
                buf0[m + k] = (int)ref[l];
                buf1[m + k] = (int)mtc[l];
            }
        }

        // Interpolate the matching signal
        if (args.m_disp_den > 1)
        {
            InterpolateLine(buf1, args.m_disp_den, args.w, args.b, args.match_interp);
            InterpolateLine(buf0, args.m_disp_den, args.w, args.b, args.match_interp);
        }

        if (args.match_interval) {
            BirchfieldTomasiMinMax(buf1, min1, max1, args.n_interp, args.b);
            if (args.match_interpolated)
                BirchfieldTomasiMinMax(buf0, min0, max0, args.n_interp, args.b);
        }

        // Compute the costs, one disparity at a time
        for (int k = 0; k < args.m_disp_n; k++)
        {
            int disp = -args.m_frame_diff_sign * (args.m_disp_den * args.disp_min + k * args.m_disp_num);

            MatchLineStruct lineArgs = {
                args.w,
                args.b,
                args.match_interpolated, //match_interpolated
                (args.match_interval) ? (args.match_interpolated) ? min0 : buf0 : buf0, // rmn
                (args.match_interval) ? (args.match_interpolated) ? max0 : buf0 : 0, // rmx
                (args.match_interval) ? min1 : buf1, // mmn
                (args.match_interval) ? max1 : 0, // mmx
                args.m_disp_n,
                disp,
                args.m_disp_den,
                args.match_fn,
                args.match_max,
                args.match_outside
            };
            
            MatchLineDevice(lineArgs, PixelAddress(m_cost, 0, y, k), &cost1[y * cost1_width]);
        }
    }
}

void LineProcess(CByteImage m_reference, CByteImage m_matching, CFloatImage m_cost, LineProcessStruct args)
{
    

    // Allocate working buffers
    BufferStruct buffs;
    int buf_length = args.n_interp * args.b; // size of one row
    int buf_size = args.h * buf_length * sizeof(int);

    AllocateGPUMemory((void**)&(buffs.buffer0), buf_size, false);
    AllocateGPUMemory((void**)&(buffs.buffer1), buf_size, false);
    AllocateGPUMemory((void**)&(buffs.min_bf0), buf_size, false);
    AllocateGPUMemory((void**)&(buffs.max_bf0), buf_size, false);
    AllocateGPUMemory((void**)&(buffs.min_bf1), buf_size, false);
    AllocateGPUMemory((void**)&(buffs.max_bf1), buf_size, false);

    // Allocate input and output image data
    uchar* m_ref_d;
    uchar* m_match_d;
    float* m_cost_d;
    float* cost1_d;

    int m_ref_size = m_reference.ImageSize();
    int m_match_size = m_matching.ImageSize();
    int m_cost_size = m_cost.ImageSize();
    int cost1_width = ((args.w - 1)*args.m_disp_den + 1);
    int cost1_size = args.h * cost1_width * sizeof(float);

    AllocateGPUMemory((void**)&m_ref_d, m_ref_size, false);
    AllocateGPUMemory((void**)&m_match_d, m_match_size, false);
    AllocateGPUMemory((void**)&m_cost_d, m_cost_size, false);
    AllocateGPUMemory((void**)&cost1_d, cost1_size, false);

    // Copy image data to device
    CopyGPUMemory(m_ref_d, m_reference.PixelAddress(0, 0, 0), m_ref_size, true);
    CopyGPUMemory(m_match_d, m_matching.PixelAddress(0, 0, 0), m_match_size, true);

    // Populate structs to hold picture info
    ImageStructUChar m_ref_struct, m_match_struct;
    ImageStructFloat m_cost_struct;

    m_ref_struct.imageSize = PopulateImageSizeStruct(m_reference);
    m_ref_struct.image = m_ref_d;
    m_match_struct.imageSize = PopulateImageSizeStruct(m_matching);
    m_match_struct.image = m_match_d;
    m_cost_struct.imageSize = PopulateImageSizeStruct(m_cost);
    m_cost_struct.image = m_cost_d;

    // Block/Grid size
    dim3 gridSize, blockSize(1, BLOCKSIZE, 1);
    gridSize.y = (unsigned int)ceil((float)(args.h) / (float)blockSize.y);

    // Kernel call
    LineProcessKernel <<<gridSize, blockSize >>>(m_ref_struct, m_match_struct, m_cost_struct, cost1_d, cost1_width, buffs, args);

    GPUERRORCHECK(cudaDeviceSynchronize());

    // Copy cost data to host
    CopyGPUMemory(m_cost.PixelAddress(0, 0, 0), m_cost_d, m_cost_size, false);

    // Free the memory
    FreeGPUMemory(buffs.buffer0);
    FreeGPUMemory(buffs.buffer1);
    FreeGPUMemory(buffs.min_bf0);
    FreeGPUMemory(buffs.max_bf0);
    FreeGPUMemory(buffs.min_bf1);
    FreeGPUMemory(buffs.max_bf1);

    FreeGPUMemory(m_ref_d);
    FreeGPUMemory(m_match_d);
    FreeGPUMemory(m_cost_d);
    FreeGPUMemory(cost1_d);
}

// Helper functions

// Return the 1-D coordinate of the band pixel
__device__ int PixelCoordToAbs(ImageSizeStruct size, int x, int y, int band)
{
    return y * size.width * size.bands + x * size.bands + band;
}

// Return pointer to the address of the specified band pixel
__device__ uchar* PixelAddress(ImageStructUChar image, int x, int y, int band)
{
    return &image.image[PixelCoordToAbs(image.imageSize, x, y, band)];
}

// Return pointer to the address of the specified band pixel
__device__ float* PixelAddress(ImageStructFloat image, int x, int y, int band)
{
    return &image.image[PixelCoordToAbs(image.imageSize, x, y, band)];
}

// Populates the ImageSizeStruct from the provided CImage
ImageSizeStruct PopulateImageSizeStruct(CImage image)
{
    ImageSizeStruct size;
    size.bands = image.Shape().nBands;
    size.height = image.Shape().height;
    size.width = image.Shape().width;
    size.bandSize = image.BandSize();
    size.pixSize = image.PixSize();
    size.rowSize = image.RowSize();

    return size;
}
