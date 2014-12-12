///////////////////////////////////////////////////////////////////////////
//
// NAME
//  StcRawCosts.cpp -- compute raw per-pixel matching score between a pair of frames
//
// DESCRIPTION
//  The CStereoMatcher class implements a number of popular stereo
//  correspondence algorithms (currently for 2-frame rectified image pairs).
//
// The definition of disparity is as follows:
//  Disparity is the (floating point) inter-frame (instantaneous) displacement
//  of a pixel between successive frames in a multi-frame stereo pair.
//
//  In building the disparity-space image (DSI) m_cost, we enumerate a set of
//  m_disp_n disparitites, k = 0..m_disp_n-1.
//
//  The mapping between the integral disparities k and the floating point
//  disparity d is given by:
//      d = disp_min + k * m_disp_num / m_disp_den
//
//  Because we may be matching frames that are not adjacent in the sequence
//  we also define the frame difference f_d and scaled disparity s_d as:
//      f_d = frame_match - frame_ref
//      s_d = f_d * d
//  The coordinate of a pixel x_m in the matching frame corresponding
//  to a reference pixel x_r in the reference frame is given by
//      x_m = x_r - s_d
//  (this is because input images are ordered left to right, so that pixel
//  motion is leftward).
//
//  When we store the floating point disparities in a gray_level image, we
//  use the formulas
//      g_d = (d - disp_min) * disp_scale
//        d =  disp_min  + g_d / disp_scale
//  to do the conversions.
//
// IMPLEMENTATION
//  For better efficiency, we first interpolate the matching line
//  up by a factor of m_disp_den.
//
// SEE ALSO
//  StereoMatcher.h         longer description of this class
//
// Copyright © Richard Szeliski, 2001.
// See Copyright.h for more details
//
///////////////////////////////////////////////////////////////////////////

#include <assert.h>
#include "Error.h"
#include "StereoMatcher.h"
#include "Convert.h"
#include "ImageIO.h"
#include "Warp1D.h"
#include <time.h>

#include "StcRawCost.h"

#define OPT1

static int gcd(int a, int b)
{
    if (b>a)
        return gcd(b, a);
    if (b==0)
        return a;
    return gcd(b, a % b);
}

void CStereoMatcher::RawCosts()
{

    float* cpu_cost = RawCostsCPU();
    float* gpu_cost = RawCostsGPU();

    VerifyComputedData(cpu_cost, gpu_cost, m_cost.ImageSize() / sizeof(float));

    free(gpu_cost);
    free(cpu_cost);

}

float* CStereoMatcher::RawCostsGPU()
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

    // cuda function call
    LineProcess(m_reference, m_matching, m_cost,
        m_disp_den, m_disp_n, b, w, h, match_interp, &match_interval, match_interpolated,
        m_frame_diff_sign, disp_min, m_disp_num, match_fn, match_max, &m_match_outside);

    PrintTiming();

    float* cost_copy = (float*)malloc(m_cost.ImageSize());
    cost_copy = (float*)memcpy(cost_copy, &m_cost.Pixel(0, 0, 0), m_cost.ImageSize());

    // Write out the different disparity images
    if (verbose >= eVerboseDumpFiles)
        WriteCosts(m_cost, "reprojected/RAW_DSI_%03d.pgm");

    return cost_copy;

}

float* CStereoMatcher::RawCostsCPU()
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

    int buffer_length = n_interp * b;

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

            MatchLine(w, b, match_interpolated,
                (match_interval) ? (match_interpolated) ? min0 : buf0 : buf0,
                (match_interval) ? (match_interpolated) ? max0 : buf0 : 0,
                (match_interval) ? min1 : buf1,
                (match_interval) ? max1 : 0,
                cost,
                m_disp_n, disp, m_disp_den,
                match_fn,
                match_max, m_match_outside);
        }
    }
    PrintTiming();

    float* cost_copy = (float*)malloc(m_cost.ImageSize());
    cost_copy = (float*)memcpy(cost_copy, &m_cost.Pixel(0, 0, 0), m_cost.ImageSize());

    // Write out the different disparity images
    if (verbose >= eVerboseDumpFiles)
        WriteCosts(m_cost, "reprojected/RAW_DSI_%03d.pgm");

    return cost_copy;
}

static void PadLine(int w, int b, float cost[],
                    int m_disp_n, int disp, int disp_den,
                    float match_outside)        // special value for outside match
{
    // Set up the starting addresses, pointers, and cutoff value
    int n = (w-1)*disp_den + 1;             // number of reference pixels
    int s = disp_den;                       // skip in reference pixels

	//  Hack: add -(s-1) to disp to make the left boundary 1 pixel wider!
	//  This is to account for possible interpolated pixels having mixed match_outside values
	//  TODO:  find a more principled solution (that also works for true occlusions)
	disp -= (s-1);

    // Fill invalid pixels
    for (int x = disp, y = 0; x < n+disp; x += s, y += b)
    {
        // Check if outside the bounds
        if (x < 0 || x >= n)
            cost[y] = match_outside;
    }
}

void CStereoMatcher::PadCosts()
{
    // Pad the matching scores with the previously computed m_match_outside value
    CShape sh = m_cost.Shape();
    int w = sh.width, h = sh.height, b = sh.nBands;

    // Process all of the lines
    for (int y = 0; y < h; y++)
    {
        // Compute the costs, one disparity at a time
        for (int k = 0; k < m_disp_n; k++)
        {
            float* cost = &m_cost.Pixel(0, y, k);
            int disp = -m_frame_diff_sign * (m_disp_den * disp_min + k * m_disp_num);
            PadLine(w, b, cost, m_disp_n, disp, m_disp_den, m_match_outside);
        }
    }
}
