// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "StereoMatcher.h"

#include <cuda.h>

#if defined(WIN32) ||  CUDAVER >= 5
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#endif

#include "StcRawCost.h"

#define BLOCKSIZE 1024

static void InterpolateLine(int buf[], int s, int w, int nB,
    EStereoInterpFn match_interp)     // interpolation function
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

void BirchfieldTomasiMinMax(const int* buffer, int* min_buf_d, int* max_buf_d, const int w, const int b)
{
    dim3 gridSize, blockSize(BLOCKSIZE, 1, 1);
    gridSize.x = (unsigned int)ceil((float)(w*b) / (float)blockSize.x);

    int* buffer_d;

    cudaMalloc(&min_buf_d, w*b*sizeof(int));
    cudaMalloc(&max_buf_d, w*b*sizeof(int));
    cudaMalloc(&buffer_d, w*b*sizeof(int));

    cudaMemcpy(buffer_d, buffer, w*b*sizeof(int), cudaMemcpyHostToDevice);

    BirchfieldTomasiMinMaxKernel<<<gridSize, blockSize>>>(buffer_d, min_buf_d, max_buf_d, w, b);
    cudaDeviceSynchronize();
}

__global__ void BirchfieldTomasiMinMaxKernel(const int* buffer, int* min_buf, int* max_buf, const int w, const int b)
{
    // Compute for every (interpolated) pixel, the minimum and maximum
    //  values in the two half-intervals before and after it
    //  (see [Birchfield & Tomasi, PAMI 20(40), April 1998, p. 401]).

    __shared__ int buffer_s[BLOCKSIZE];
    __shared__ int min_buf_s[BLOCKSIZE];
    __shared__ int max_buf_s[BLOCKSIZE];

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
            min_buf_s[l] = __min(Il, __min(b0, Ir));
            max_buf_s[l] = __max(Il, __max(b0, Ir));
        }
    }
    __syncthreads();

    if (k < b)
    {
        for (int x = 0, l = k; x < w; x++, l += b)
        {
            min_buf[l] = min_buf_s[l];
            max_buf[l] = max_buf_s[l];
        }
    }
}

#define BAD_COST -1

static bool undefined_cost = true;     // set this to true to pad with outside_cost

void MatchLine(int w, int b, int interpolated,
    int rmn[], int rmx[],     // min/max of ref (ref if rmx == 0)
    int mmn[], int mmx[],     // min/max of mtc (mtc if mmx == 0)
    float cost[],
    int m_disp_n, int disp, int disp_den,
    EStereoMatchFn match_fn,  // matching function
    int match_max,            // maximum difference for truncated SAD/SSD
    float match_outside,        // special value for outside match
    int match_interval,
    int match_interpolated)
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

    MatchPixels(w, b, interpolated, rmn, rmx, mmn, mmx, (float*)&cost1[0], disp, match_fn, n, s, cutoff, match_interval, match_interpolated);

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
    if (undefined_cost)
        left_cost = right_cost = match_outside;

    for (int x = 0; x < n && cost1[x] == bad_cost; x += s)
        cost1[x] = left_cost;
    for (int x = n - 1; x >= 0 && cost1[x] == bad_cost; x -= s)
        cost1[x] = right_cost;

    // Box filter if interpolated costs
    BoxFilter((float*)&cost1[0], cost, n, w, m_disp_n, disp_den, interpolated);
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
    int match_interpolated)
{
    dim3 gridSize, blockSize(BLOCKSIZE, 1, 1);
    gridSize.x = (unsigned int)ceil((float)(w*b) / (float)blockSize.x);

    float* cost1_d;
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
    cudaMalloc(&cost1_d, n*sizeof(float));

    // Input arrays host/device location is conditional
    if (match_interval)
    {
        if (match_interpolated)
        {
            // min0 and max0 already on device from Birchfield
            rmn_d = rmn;
            rmx_d = rmx;
        }
        else
        {
            // Birchfield with buf0 not run on device
            cudaMemcpy(rmn_d, rmn, w*b*sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(rmx_d, rmx, w*b*sizeof(int), cudaMemcpyHostToDevice);
        }
        // min1 and max1 already on device from Birchfield
        mmn_d = mmn;
        mmx_d = mmx;
    }
    else
    {
        // Birchfield not run, so buf0 and buf1 not on device
        cudaMemcpy(rmn_d, rmn, w*b*sizeof(int), cudaMemcpyHostToDevice);
        rmx_d = rmx; // null
        cudaMemcpy(mmn_d, mmn, w*b*sizeof(int), cudaMemcpyHostToDevice);
        mmx_d = mmx; // null
    }

    MatchPixelsKernel<<<gridSize, blockSize>>>(w, b, interpolated, rmn, rmx, mmn, mmx, cost1_d, disp, match_fn, n, s, cutoff);
    cudaDeviceSynchronize();

    // Copy output back to host
    cudaMemcpy(cost1, cost1_d, w*b*sizeof(float), cudaMemcpyDeviceToHost);
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

__global__ void MatchPixelsKernel(int w, int b, int interpolated,
                                    int* rmn, int* rmx,     // min/max of ref (ref if rmx == 0)
                                    int* mmn, int* mmx,     // min/max of mtc (mtc if mmx == 0)
                                    float* cost1,
                                    int disp,
                                    EStereoMatchFn match_fn,  // matching function
                                    int n,
                                    int s,
                                    int cutoff)
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
            __syncthreads();
        }
    }
}

void BoxFilter(float* cost1, float* cost, int n, int w, int m_disp_n, int disp_den, int interpolated)
{
    dim3 gridSize, blockSize(BLOCKSIZE, BLOCKSIZE, 1);
    gridSize.x = (unsigned int)ceil((float)(n) / (float)blockSize.x);
    gridSize.y = (unsigned int)ceil((float)(w*m_disp_n) / (float)blockSize.y);

    float* cost_d;
    float* cost1_d;
    cudaMalloc(&cost_d, w*m_disp_n*sizeof(float));
    cudaMalloc(&cost1_d, n*sizeof(float));
    cudaMemcpy(cost1_d, cost1, n*sizeof(float), cudaMemcpyHostToDevice);

    BoxFilterKernel<<<gridSize, blockSize>>>(cost1, cost, n, w, m_disp_n, disp_den, interpolated);
    cudaDeviceSynchronize();

    cudaMemcpy(cost, cost_d, w*m_disp_n*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(cost_d);
    cudaFree(cost1_d);
}

__global__ void BoxFilterKernel(float* cost1, float* cost, int n, int w, int m_disp_n, int disp_den, int interpolated)
{
    __shared__ float cost1_s[BLOCKSIZE];

    unsigned x = (threadIdx.x + blockIdx.x * blockDim.x) * disp_den;
    unsigned y = (threadIdx.y + blockIdx.y * blockDim.y) * m_disp_n;

    if (x < n)
        cost1_s[x] = cost1[x];
    __syncthreads();

    // Box filter if interpolated costs
    int dh = disp_den / 2;
    float box_scale = 1.0 / (2 * dh + 1);
    if (y < w*m_disp_n)
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

void CStereoMatcher::RawCosts()
{
    StartTiming();

    // Compute raw per-pixel matching score between a pair of frames
    CShape sh = m_reference.Shape();
    int w = sh.width, h = sh.height, b = sh.nBands;

    if (verbose >= eVerboseProgress)
        fprintf(stderr, "- computing costs: ");
    if (verbose >= eVerboseSummary) {
        fprintf(stderr, match_fn == eAD ? "AD" : (match_fn == eSD ? "SD" : "???"));
        if (m_disp_step != 1.0f)
            fprintf(stderr, ", step=%g", m_disp_step);
        if (match_max < 1000)
            fprintf(stderr, ", trunc=%d", match_max);
        if (match_interval)
            fprintf(stderr, ", interval");
        if (match_interpolated)
            fprintf(stderr, ", interpolated");
    }
    if (verbose >= eVerboseProgress)
        fprintf(stderr, "\n");

    // Allocate a buffer for interpolated values
    //  Note that we don't have to interpolate the ref image if we
    //  aren't using match_interpolated, but it's simpler to code this way.
    match_interval = (match_interval ? 1 : 0);  // force to [0,1]
    int n_interp = m_disp_den * (w - 1) + 1;
    std::vector<int> buffer0, buffer1, min_bf0, max_bf0, min_bf1, max_bf1;
    buffer0.resize(n_interp * b);
    buffer1.resize(n_interp * b);
    min_bf0.resize(n_interp * b);
    max_bf0.resize(n_interp * b);
    min_bf1.resize(n_interp * b);
    max_bf1.resize(n_interp * b);

    // Special value for border matches
    int worst_match = b * ((match_fn == eSD) ? 255 * 255 : 255);
    int cutoff = (match_fn == eSD) ? match_max * match_max : abs(match_max);
    m_match_outside = __min(worst_match, cutoff);	// trim to cutoff

    // Process all of the lines
    for (int y = 0; y < h; y++)
    {
        uchar* ref = &m_reference.Pixel(0, y, 0);
        uchar* mtc = &m_matching.Pixel(0, y, 0);
        int*  buf0 = &buffer0[0];
        int*  buf1 = &buffer1[0];
        int*  min0 = &min_bf0[0];
        int*  max0 = &max_bf0[0];
        int*  min1 = &min_bf1[0];
        int*  max1 = &max_bf1[0];

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

        // Parallelized
        if (match_interval) {
            BirchfieldTomasiMinMax(buf1, min1, max1, n_interp, b);
            if (match_interpolated)
                BirchfieldTomasiMinMax(buf0, min0, max0, n_interp, b);
        }

        // Compute the costs, one disparity at a time
        for (int k = 0; k < m_disp_n; k++)
        {
            float* cost = &m_cost.Pixel(0, y, k);
            int disp = -m_frame_diff_sign * (m_disp_den * disp_min + k * m_disp_num);
            // Parallelized
            MatchLine(w, b, match_interpolated,
                (match_interval) ? (match_interpolated) ? min0 : buf0 : buf0,
                (match_interval) ? (match_interpolated) ? max0 : buf0 : 0,
                (match_interval) ? min1 : buf1,
                (match_interval) ? max1 : 0,
                cost, m_disp_n, disp, m_disp_den,
                match_fn, match_max, m_match_outside);
        }
    }
    PrintTiming();

    // Write out the different disparity images
    if (verbose >= eVerboseDumpFiles)
        WriteCosts(m_cost, "reprojected/RAW_DSI_%03d.pgm");
}