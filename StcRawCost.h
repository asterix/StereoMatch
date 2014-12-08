
#include "StereoParameters.h"

#ifndef STCRAWCOST
#define STCRAWCOST

#define BLOCKSIZE 1024
#define BAD_COST -1

static bool undefined_cost = true;     // set this to true to pad with outside_cost

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

#endif
