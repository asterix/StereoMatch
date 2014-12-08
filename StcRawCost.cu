// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "StereoMatcher.h"
#include "Warp1D.h"

#include <cuda.h>

#if defined(WIN32) ||  CUDAVER >= 5
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#endif

#include "StcRawCost.h"

// Serial Execution Versions

__device__ __host__ void InterpolateLine(int buf[], int s, int w, int nB, EStereoInterpFn match_interp)     // interpolation function
{
    // Interpolate the missing values
    float si = 1.0f / s;
    for (int x = 0; x < w - 1; x++)
    {
        for (int b = 0; b < nB; b++)
        {
            int *v = &buf[s*x*nB + b];
            float I0 = v[0];
            float I1 = v[s*nB];
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
                    float Ii = CubicInterpolate(sf, Im, I0, I1, Ip);
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

__device__ __host__ void BirchfieldTomasiMinMax(const int* buffer, int* min_buf, int* max_buf, const int w, const int b)
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

__device__ __host__ void MatchLine(int w, int b, int interpolated,
    int rmn[], int rmx[],     // min/max of ref (ref if rmx == 0)
    int mmn[], int mmx[],     // min/max of mtc (mtc if mmx == 0)
    float cost[],
    int m_disp_n, int disp, int disp_den,
    EStereoMatchFn match_fn,  // matching function
    int match_max,            // maximum difference for truncated SAD/SSD
    float match_outside)        // special value for outside match
{
    // Set up the starting addresses, pointers, and cutoff value
    int n = (w - 1)*disp_den + 1;             // number of reference pixels
    int s = (interpolated) ? 1 : disp_den;     // skip in reference pixels
    std::vector<float> cost1;
    cost1.resize(n);
    int cutoff = (match_fn == eSD) ? match_max * match_max : abs(match_max);
    // TODO:  cutoff is not adjusted for the number of bands...
    const float bad_cost = -1;

    // Match valid pixels
    float  left_cost = bad_cost;
    float right_cost = bad_cost;
    int x, y;
    for (x = 0; x < n; x += s)
    {
        // Compute ref and match pointers
        cost1[x] = bad_cost;
        int x_r = x, x_m = x + disp;
        if (x_m < 0 || x_m >= n)
            continue;
        int* rn = &rmn[x_r*b];    // pointer to ref or min pixel(s)
        int* rx = &rmx[x_r*b];    // pointer to ref    max pixel(s)
        int* mn = &mmn[x_m*b];    // pointer to mtc or min pixel(s)
        int* mx = &mmx[x_m*b];    // pointer to mtc    max pixel(s)
        int  diff_sum = 0;        // accumulated error

        // This code could be special-cased for b==1 for more efficiency...
        for (int ib = 0; ib < b; ib++)
        {
            int diff1 = mn[ib] - rn[ib];    // straightforward difference
            if (rmx && mmx)
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
            int diff2 = (match_fn == eSD) ?    // squared or absolute difference
                diff1 * diff1 : abs(diff1);
            diff_sum += diff2;
        }
        int diff3 = __min(diff_sum, cutoff);    // truncated difference
        if (left_cost == bad_cost)
            left_cost = diff3;  // first cost computed
        right_cost = diff3;     // last  cost computed
        cost1[x] = diff3;        // store in temporary array
    }

    // Fill in the left and right edges
    if (UNDEFINED_COST)
        left_cost = right_cost = match_outside;
    for (x = 0; x < n && cost1[x] == bad_cost; x += s)
        cost1[x] = left_cost;
    for (x = n - 1; x >= 0 && cost1[x] == bad_cost; x -= s)
        cost1[x] = right_cost;

    // Box filter if interpolated costs
    int dh = disp_den / 2;
    float box_scale = 1.0 / (2 * dh + 1);
    for (x = 0, y = 0; y < w*m_disp_n; x += disp_den, y += m_disp_n)
    {
        if (interpolated && disp_den > 1)
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

// Parallel Execution Versions

__global__ void BirchfieldTomasiMinMaxKernel(const int* buffer, int* min_buf, int* max_buf, const int w, const int b, int buffer_length)
{
    // Compute for every (interpolated) pixel, the minimum and maximum
    //  values in the two half-intervals before and after it
    //  (see [Birchfield & Tomasi, PAMI 20(40), April 1998, p. 401]).

    __shared__ int buffer_s[BLOCKSIZE];

    unsigned k = threadIdx.x + blockIdx.x * blockDim.x;
    //unsigned x = threadIdx.y + blockIdx.y * blockDim.y;

    if (k < b)
    {
        for (int x = 0, l = k; x < w; x++, l += b)
        {
            buffer_s[l] = buffer[l];
        }
    }
    __syncthreads();

    // Process each band separately
    if (k < b)
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

void BirchfieldTomasiMinMax(const int* buffer, int* min_buf_d, int* max_buf_d, const int w, const int b, int buffer_length)
{
    dim3 gridSize, blockSize(BLOCKSIZE, 1, 1);
    gridSize.x = (unsigned int)ceil((float)(w*b) / (float)blockSize.x);

    int* buffer_d;

    cudaMalloc(&min_buf_d, buffer_length*sizeof(int));
    cudaMalloc(&max_buf_d, buffer_length*sizeof(int));
    cudaMalloc(&buffer_d, buffer_length*sizeof(int));

    cudaMemcpy(buffer_d, buffer, w*b*sizeof(int), cudaMemcpyHostToDevice);

    BirchfieldTomasiMinMaxKernel<<<gridSize, blockSize>>>(buffer_d, min_buf_d, max_buf_d, w, b, buffer_length);
    cudaDeviceSynchronize();

    // Don't copy anything back since next call to MatchLine is parallelized
}


void MatchLine(int w, int b, int interpolated,
    int* rmn, int* rmx,     // min/max of ref (ref if rmx == 0)
    int* mmn, int* mmx,     // min/max of mtc (mtc if mmx == 0)
    float* cost,
    int m_disp_n, int disp, int disp_den,
    EStereoMatchFn match_fn,  // matching function
    int match_max,            // maximum difference for truncated SAD/SSD
    float match_outside,        // special value for outside match
    int match_interval,
    int match_interpolated,
    int buffer_length) // length of the rmn, rmx, mmn, mmx buffers
{
    // Set up the starting addresses, pointers, and cutoff value
    int n = (w - 1)*disp_den + 1;             // number of reference pixels
    int s = (interpolated) ? 1 : disp_den;     // skip in reference pixels
    float* cost1 = (float*)malloc(n*sizeof(float));
    int cutoff = (match_fn == eSD) ? match_max * match_max : abs(match_max);
    // TODO:  cutoff is not adjusted for the number of bands...
    const float bad_cost = -1;

    int cost1_length = n;
    int cost_length = w * m_disp_n;

    // Match valid pixels
    float  left_cost = bad_cost;
    float right_cost = bad_cost;

    MatchPixels(w, b, interpolated, rmn, rmx, mmn, mmx, cost1, disp, match_fn, n, s, cutoff, match_interval, match_interpolated, buffer_length, cost1_length);

    // left & right cost search
    for (int x = 0; x < n; x++)
    {
        if (cost1[x] != bad_cost)
        {
            left_cost = cost1[x];
            break;
        }
    }
    for (int x = n - 1; x >= 0; x--)
    {
        if (cost1[x] != bad_cost)
        {
            right_cost = cost1[x];
            break;
        }
    }

    // Fill in the left and right edges
    if (UNDEFINED_COST)
        left_cost = right_cost = match_outside;

    for (int x = 0; x < cost1_length && cost1[x] == bad_cost; x += s)
        cost1[x] = left_cost;
    for (int x = cost1_length - 1; x >= 0 && cost1[x] == bad_cost; x -= s)
        cost1[x] = right_cost;

    // Box filter if interpolated costs
    BoxFilter(cost1, cost, n, w, m_disp_n, disp_den, interpolated, cost1_length, cost_length);
}


__global__ void MatchPixelsKernel(int w, int b, int interpolated,
    int* rmn, int* rmx,     // min/max of ref (ref if rmx == 0)
    int* mmn, int* mmx,     // min/max of mtc (mtc if mmx == 0)
    float* cost1,
    int disp,
    EStereoMatchFn match_fn,  // matching function
    int n,
    int s,
    int cutoff,
    int buffer_length,
    int cost1_length)
{

    unsigned x = (threadIdx.x + blockIdx.x * blockDim.x) * s;

    // Match valid pixels
    if (x < n)
    {
        // Compute ref and match pointers
        cost1[x] = BAD_COST;
        int x_r = x;
        int x_m = x + disp;
        if (x_m >= 0 && x_m < n)
        {
            int* rn = &rmn[x_r*b];    // pointer to ref or min pixel(s)
            int* rx = &rmx[x_r*b];    // pointer to ref    max pixel(s)
            int* mn = &mmn[x_m*b];    // pointer to mtc or min pixel(s)
            int* mx = &mmx[x_m*b];    // pointer to mtc    max pixel(s)
            int  diff_sum = 0;        // accumulated error

            // This code could be special-cased for b==1 for more efficiency...
            for (int ib = 0; ib < b; ib++)
            {
                int diff1 = mn[ib] - rn[ib];    // straightforward difference
                if (rmx && mmx)
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
                int diff2 = (match_fn == eSD) ?    // squared or absolute difference
                    diff1 * diff1 : abs(diff1);
                diff_sum += diff2;
            }
            int diff3 = __min(diff_sum, cutoff);    // truncated difference
            __syncthreads();

            cost1[x] = diff3;        // store in temporary array
        }
    }
}

void MatchPixels(int w, int b, int interpolated,
    int* rmn, int* rmx,     // min/max of ref (ref if rmx == 0)
    int* mmn, int* mmx,     // min/max of mtc (mtc if mmx == 0)
    float* cost1,
    int disp,
    EStereoMatchFn match_fn,  // matching function
    int n,
    int s,
    int cutoff,
    int match_interval,
    int match_interpolated,
    int buffer_length,
    int cost1_length)
{
    dim3 gridSize, blockSize(BLOCKSIZE, 1, 1);
    gridSize.x = (unsigned int)ceil((float)(cost1_length) / (float)blockSize.x);

    float* cost1_d; // length n
    int* rmn_d;
    int* rmx_d;
    int* mmn_d;
    int* mmx_d;

    /*
    rmn = (match_interval) ? (match_interpolated) ? min0 : buf0 : buf0,
    rmx = (match_interval) ? (match_interpolated) ? max0 : buf0 : 0,
    mmn = (match_interval) ? min1 : buf1,
    mmx = (match_interval) ? max1 : 0,
    */

    // cost1 is the output and is not on device
    cudaMalloc(&cost1_d, cost1_length*sizeof(float));

    // Input arrays host/device location is conditional
    if (match_interval)
    {
        if (match_interpolated)
        {
            // rmn = min0 and rmx = max0 already on device from Birchfield
            rmn_d = rmn;
            rmx_d = rmx;
        }
        else
        {
            // Birchfield with rmn = rmx = buf0 not run on device
            cudaMalloc(&rmn_d, buffer_length*sizeof(int));
            cudaMemcpy(rmn_d, rmn, buffer_length*sizeof(int), cudaMemcpyHostToDevice);
            cudaMalloc(&rmx_d, buffer_length*sizeof(int));
            cudaMemcpy(rmx_d, rmx, buffer_length*sizeof(int), cudaMemcpyHostToDevice);
        }
        // min1 and max1 already on device from Birchfield
        mmn_d = mmn;
        mmx_d = mmx;
    }
    else
    {
        // Birchfield not run, so buf0 and buf1 not on device
        cudaMalloc(&rmn_d, buffer_length*sizeof(int));
        cudaMemcpy(rmn_d, rmn, buffer_length*sizeof(int), cudaMemcpyHostToDevice);
        rmx_d = rmx; // null
        cudaMalloc(&mmn_d, buffer_length*sizeof(int));
        cudaMemcpy(mmn_d, mmn, buffer_length*sizeof(int), cudaMemcpyHostToDevice);
        mmx_d = mmx; // null
    }

    MatchPixelsKernel<<<gridSize, blockSize>>>(w, b, interpolated, rmn_d, rmx_d, mmn_d, mmx_d, cost1_d, disp, match_fn, n, s, cutoff, buffer_length, cost1_length);
    cudaDeviceSynchronize();

    // Copy output back to host
    cudaMemcpy(cost1, cost1_d, cost1_length*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(cost1_d);

    // Free device memory
    if (match_interval)
    {
        cudaFree(rmn_d);
        cudaFree(rmx_d);
        cudaFree(mmn_d);
        cudaFree(mmx_d);
    }
    else
    {
        // Birchfield not run, so buf0 and buf1 not on device
        cudaFree(rmn_d);
        // rmx_d null
        cudaFree(mmn_d);
        // mmx_d null
    }
}


__global__ void BoxFilterKernel(float* cost1, float* cost, int n, int w, int m_disp_n, int disp_den, int interpolated, int cost1_length, int cost_length)
{
    __shared__ float cost1_s[BLOCKSIZE];

    unsigned x = (threadIdx.x + blockIdx.x * blockDim.x) * disp_den;
    unsigned y = (threadIdx.y + blockIdx.y * blockDim.y) * m_disp_n;

    if (x < cost1_length)
        cost1_s[x] = cost1[x];
    __syncthreads();

    // Box filter if interpolated costs
    int dh = disp_den / 2;
    float box_scale = 1.0 / (2 * dh + 1);
    if (y < cost_length && x < cost1_length)
    {
        if (interpolated && disp_den > 1)
        {
            float sum = 0;
            for (int k = -dh; k <= dh; k++)
            {
                int l = __max(0, __min(n - 1, x + k));  // TODO: make more efficient
                sum += cost1_s[l];
            }
            cost[y] = int(box_scale * sum + 0.5);
        }
        else
            cost[y] = cost1_s[x];
    }
}

void BoxFilter(float* cost1, float* cost, int n, int w, int m_disp_n, int disp_den, int interpolated, int cost1_length, int cost_length)
{
    dim3 gridSize, blockSize(BLOCKSIZE, BLOCKSIZE, 1);
    gridSize.x = (unsigned int)ceil((float)(cost1_length) / (float)blockSize.x);
    gridSize.y = (unsigned int)ceil((float)(cost_length) / (float)blockSize.y);

    float* cost_d; // cost_length
    float* cost1_d; // cost1_length

    cudaMalloc(&cost_d, cost_length*sizeof(float));
    cudaMalloc(&cost1_d, cost1_length*sizeof(float));

    cudaMemcpy(cost1_d, cost1, cost1_length*sizeof(float), cudaMemcpyHostToDevice);

    BoxFilterKernel<<<gridSize, blockSize>>>(cost1_d, cost_d, n, w, m_disp_n, disp_den, interpolated, cost1_length, cost_length);
    cudaDeviceSynchronize();

    cudaMemcpy(cost, cost_d, cost_length*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(cost_d);
    cudaFree(cost1_d);
}

// Parallelized Execution 2

__device__ int PixelCoordToAbs(ImageSizeStruct size, int x, int y, int band)
{
    return y * size.rowSize + x * size.pixSize + band * size.bandSize;
}

__global__ void LineProcessKernel(ImageStructUChar m_reference, ImageStructUChar m_matching, ImageStructFloat m_cost,
    int* buffer0, int* buffer1, int* min_bf0, int* max_bf0, int* min_bf1, int* max_bf1,
    int m_disp_den, int m_disp_n, int b, int w, int h, EStereoInterpFn match_interp, int match_interval, int match_interpolated, int n_interp,
    int m_frame_diff_sign, int disp_min, int m_disp_num, EStereoMatchFn match_fn, int match_max, float match_outside)
{
    unsigned y = (threadIdx.y + blockIdx.y * blockDim.y);
    unsigned buf_start = y * m_disp_den * b;

    // Process all of the lines
    if (y < h)
    {
        uchar* ref = &m_reference.image[PixelCoordToAbs(m_reference.imageSize, 0, y, 0)];
        uchar* mtc = &m_matching.image[PixelCoordToAbs(m_matching.imageSize, 0, y, 0)];
        int*  buf0 = &buffer0[buf_start];
        int*  buf1 = &buffer1[buf_start];
        int*  min0 = &min_bf0[buf_start];
        int*  max0 = &max_bf0[buf_start];
        int*  min1 = &min_bf1[buf_start];
        int*  max1 = &max_bf1[buf_start];

        // Fill the line buffers
        int x, l, m;
        for (x = 0, l = 0, m = 0; x < w; x++, m += m_disp_den*b)
        {
            for (int k = 0; k < b; k++, l++)
            {
                buf0[m + k] = ref[l];
                buf1[m + k] = mtc[l];
            }
        }

        // Interpolate the matching signal
        if (m_disp_den > 1)
        {
            InterpolateLine(buf1, m_disp_den, w, b, match_interp);
            InterpolateLine(buf0, m_disp_den, w, b, match_interp);
        }

        if (match_interval) {
            BirchfieldTomasiMinMax(buf1, min1, max1, n_interp, b);
            if (match_interpolated)
                BirchfieldTomasiMinMax(buf0, min0, max0, n_interp, b);
        }

        // Compute the costs, one disparity at a time
        for (int k = 0; k < m_disp_n; k++)
        {
            float* cost = &m_cost.image[PixelCoordToAbs(m_cost.imageSize, 0, y, k)];
            int disp = -m_frame_diff_sign * (m_disp_den * disp_min + k * m_disp_num);

            MatchLine(w, b, match_interpolated,
                (match_interval) ? (match_interpolated) ? min0 : buf0 : buf0,
                (match_interval) ? (match_interpolated) ? max0 : buf0 : 0,
                (match_interval) ? min1 : buf1,
                (match_interval) ? max1 : 0,
                cost,
                m_disp_n, disp, m_disp_den,
                match_fn,
                match_max,
                match_outside);
        }
    }
}

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

void LineProcess(CByteImage m_reference, CByteImage m_matching, CFloatImage m_cost,
    int m_disp_den, int m_disp_n, int b, int w, int h, EStereoInterpFn match_interp, int* match_interval, int match_interpolated,
    int m_frame_diff_sign, int disp_min, int m_disp_num, EStereoMatchFn match_fn, int match_max, float* m_match_outside)
{
    // Allocate a buffer for interpolated values
    //  Note that we don't have to interpolate the ref image if we
    //  aren't using match_interpolated, but it's simpler to code this way.
    *match_interval = (match_interval ? 1 : 0);  // force to [0,1]
    int n_interp = m_disp_den * (w - 1) + 1;

    // Allocate working buffers
    iptr buffer0, buffer1, min_bf0, max_bf0, min_bf1, max_bf1;
    int buf_length = n_interp * b;
    int buf_size = h * buf_length * sizeof(int);
    
    cudaMalloc(&buffer0, buf_size);
    cudaMalloc(&buffer1, buf_size);
    cudaMalloc(&min_bf0, buf_size);
    cudaMalloc(&max_bf0, buf_size);
    cudaMalloc(&min_bf1, buf_size);
    cudaMalloc(&max_bf1, buf_size);

    // Allocate input and output image data
    uchar *m_ref_d, *m_match_d;
    float* m_cost_d;

    int m_ref_size = m_reference.ImageSize() * sizeof(uchar);
    int m_match_size = m_matching.ImageSize() * sizeof(uchar);
    int m_cost_size = m_cost.ImageSize() * sizeof(float);

    cudaMalloc(&m_ref_d, m_ref_size);
    cudaMalloc(&m_match_d, m_match_size);
    cudaMalloc(&m_cost_d, m_cost_size);

    // Copy image data to device
    cudaMemcpy(m_ref_d, &m_reference.Pixel(0, 0, 0), m_ref_size, cudaMemcpyHostToDevice);
    cudaMemcpy(m_match_d, &m_matching.Pixel(0, 0, 0), m_match_size, cudaMemcpyHostToDevice);

    // Populate structs to hold picture info
    ImageStructUChar m_ref_struct, m_match_struct;
    ImageStructFloat m_cost_struct;

    m_ref_struct.imageSize = PopulateImageSizeStruct(m_reference);
    m_ref_struct.image = m_ref_d;
    m_match_struct.imageSize = PopulateImageSizeStruct(m_matching);
    m_match_struct.image = m_match_d;
    m_cost_struct.imageSize = PopulateImageSizeStruct(m_cost);
    m_cost_struct.image = m_cost_d;

    // Special value for border matches
    int worst_match = b * ((match_fn == eSD) ? 255 * 255 : 255);
    int cutoff = (match_fn == eSD) ? match_max * match_max : abs(match_max);
    *m_match_outside = __min(worst_match, cutoff);	// trim to cutoff

    dim3 gridSize, blockSize(1, BLOCKSIZE, 1);
    gridSize.y = (unsigned int)ceil((float)(h) / (float)blockSize.y);
    
    // Kernel call
    LineProcessKernel<<<gridSize, blockSize>>>(m_ref_struct, m_match_struct, m_cost_struct,
        buffer0, buffer1, min_bf0, max_bf0, min_bf1, max_bf1,
        m_disp_den, m_disp_n, b, w, h, match_interp, *match_interval, match_interpolated, n_interp,
        m_frame_diff_sign, disp_min, m_disp_num, match_fn, match_max, *m_match_outside);

    cudaDeviceSynchronize();

    // Copy cost data to host
    cudaMemcpy(&m_cost.Pixel(0, 0, 0), m_cost_d, m_cost_size, cudaMemcpyDeviceToHost);

    // Free the memory
    cudaFree(buffer0);
    cudaFree(buffer1);
    cudaFree(min_bf0);
    cudaFree(max_bf0);
    cudaFree(min_bf1);
    cudaFree(max_bf1);

    cudaFree(m_ref_d);
    cudaFree(m_match_d);
    cudaFree(m_cost_d);
}