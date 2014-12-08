
#include "StereoParameters.h"

#ifndef STCRAWCOST_H
#define STCRAWCOST_H

#define BLOCKSIZE 1024
#define BAD_COST -1

#define UNDEFINED_COST true // set this to true to pad with outside_cost

typedef int* iptr;

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

void InterpolateLine(int buf[], int s, int w, int nB, EStereoInterpFn match_interp);
void BirchfieldTomasiMinMax(const int* buffer, int* min_buf, int* max_buf, const int w, const int b);
void MatchLine(int w, int b, int interpolated,
    int rmn[], int rmx[],     // min/max of ref (ref if rmx == 0)
    int mmn[], int mmx[],     // min/max of mtc (mtc if mmx == 0)
    float cost[],
    int m_disp_n, int disp, int disp_den,
    EStereoMatchFn match_fn,  // matching function
    int match_max,            // maximum difference for truncated SAD/SSD
    float match_outside);

void BirchfieldTomasiMinMax(const int* buffer, int* min_buf_d, int* max_buf_d, const int w, const int b, int buffer_length);

void MatchLine(int w, int b, int interpolated,
    int rmn[], int rmx[],     // min/max of ref (ref if rmx == 0)
    int mmn[], int mmx[],     // min/max of mtc (mtc if mmx == 0)
    float cost[],
    int m_disp_n, int disp, int disp_den,
    EStereoMatchFn match_fn,  // matching function
    int match_max,            // maximum difference for truncated SAD/SSD
    float match_outside,        // special value for outside match
    int match_interval,
    int match_interpolated,
    int buffer_length);

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
    int cost1_length);

void BoxFilter(float* cost1, float* cost, int n, int w, int m_disp_n, int disp_den, int interpolated, int cost1_length, int cost_length);

void LineProcess(CByteImage m_reference, CByteImage m_matching, CFloatImage m_cost,
    int m_disp_den, int m_disp_n, int b, int w, int h, EStereoInterpFn match_interp, int* match_interval, int match_interpolated,
    int m_frame_diff_sign, int disp_min, int m_disp_num, EStereoMatchFn match_fn, int match_max, float* m_match_outside);

#endif
