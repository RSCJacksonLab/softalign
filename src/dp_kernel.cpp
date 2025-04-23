#include "softalign.hpp"
#include <immintrin.h>
#include <algorithm>
#include <cmath>
#include <omp.h>
#include <cstdint> 

namespace sa {
namespace {

inline float hsum(__m256 v) {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 sum = _mm_add_ps(hi, lo);
    sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
    sum = _mm_add_ss(sum, _mm_shuffle_ps(sum, sum, 1));
    return _mm_cvtss_f32(sum);
}

/* hybrid JS + BLOSUM distance for one column pair */
float hybrid_score(const float* p,
                   const float* q,
                   const float* M,
                   float alpha)
{
    /* BLOSUM part */
    __m256 acc_blos = _mm256_setzero_ps();
    for (int k = 0; k < 16; k += 8) {
        __m256 pv = _mm256_loadu_ps(p + k);
        __m256 qv = _mm256_loadu_ps(q + k);
        __m256 mv = _mm256_loadu_ps(M + k*20);          // rough but fine
        acc_blos  = _mm256_fmadd_ps(pv, _mm256_mul_ps(mv,qv), acc_blos);
    }
    float blos_tail = 0.f;
    for (int k = 16; k < 20; ++k)
        blos_tail += p[k] * M[k*20 + k] * q[k];
    float blosum = hsum(acc_blos) + blos_tail;
    float blos_dist = 1.f - blosum;                    // already /max(M)

    /* JS part */
    constexpr float EPS = 1e-8f;               // small positive number
    float kl1 = 0.f, kl2 = 0.f;
    for (int k = 0; k < 20; ++k) {
        float pk = p[k] + EPS;                 // avoid log(0)
        float qk = q[k] + EPS;
        float m  = 0.5f * (pk + qk);
        kl1 += pk * std::log(pk / m);
        kl2 += qk * std::log(qk / m);
    }
    float js = std::sqrt(0.5f * (kl1 + kl2));

    return alpha*js + (1.f-alpha)*blos_dist;
}

constexpr float NEG_INF = -1e30f;

/* helper to copy a 20‑float column + gap flag */
inline void push_col(std::vector<f32>& dst, const f32* src, bool gap=false)
{
    if (gap) {
        dst.insert(dst.end(), 20, 0.f);
        dst.push_back(1.f);
    } else {
        dst.insert(dst.end(), src, src+20);
        dst.push_back(0.f);
    }
}

} 

AlignmentResult
nw_affine(const ProbSeq& a, const ProbSeq& b, const SubstMat& M,
          float gap_open, float gap_ext, float alpha)
{
    const int L1 = a.L, L2 = b.L;
                      
    /* rolling DP rows */
    std::vector<f32> Mp(L2+1,  NEG_INF), Xp(L2+1, NEG_INF), Yp(L2+1, NEG_INF),
                     Mc(L2+1), Xc(L2+1), Yc(L2+1);
    Mp[0] = 0.f;

    for (int j = 1; j <= L2; ++j) {
        Mp[j] = NEG_INF;
        Xp[j] = NEG_INF;
        Yp[j] = -gap_open - (j-1) * gap_ext;    // cost of j-column gap in seq-A
    }

    const int band = std::max(L1, L2);                 // no banding
    std::vector<uint8_t> TB((L1+1)*(L2+1), 0);         // full (L1+1)×(L2+1) grid
    auto idx = [&](int i, int j){ return i*(L2+1) + j; };

    for (int i = 1; i <= L1; ++i) { // gap in seq-B on left edge.
        Mc[0]=NEG_INF;
        Xc[0]=-gap_open-(i-1)*gap_ext;
        Yc[0]=NEG_INF;

        for (int j = 1; j <= L2; ++j) {        // full width
            float s = -hybrid_score(a.row(i-1), b.row(j-1), M.data(), alpha);
    
            /* M */
            float m = Mp[j-1] + s;
            float x = Xp[j-1] + s;
            float y = Yp[j-1] + s;
            Mc[j] = std::max({m, x, y});
            TB[idx(i, j)] = (Mc[j] == m) ? 0 : (Mc[j] == x ? 1 : 2);
    
            /* X (gap in B) */
            Xc[j] = std::max(Mp[j] - gap_open, Xp[j] - gap_ext);
    
            /* Y (gap in A) – depends on Yc[j-1] written this iteration */
            Yc[j] = std::max(Mc[j-1] - gap_open, Yc[j-1] - gap_ext);
        }
        std::swap(Mp,Mc); std::swap(Xp,Xc); std::swap(Yp,Yc);
    }

    /* traceback  */
    std::vector<f32> bufA, bufB;
    bufA.reserve((L1+L2)*21); bufB.reserve((L1+L2)*21);

    int i=L1, j=L2; uint8_t state = TB[idx(i,j)];
    if (Mp[L2] >= Xp[L2] && Mp[L2] >= Yp[L2]) state=0;
    else if (Xp[L2] >= Yp[L2]) state=1;
    else state=2;

    while (i>0 || j>0) {
        if (state==0) {
            push_col(bufA, a.row(i-1));
            push_col(bufB, b.row(j-1));
            state = TB[idx(i,j)];
            --i; --j;
        } else if (state==1) {                 // gap in b
            push_col(bufA, a.row(i-1));
            push_col(bufB, nullptr, true);
            state = (Mp[j]-gap_open > Xp[j]-gap_ext)?0:1;
            --i;
        } else {                               // gap in a
            push_col(bufA, nullptr, true);
            push_col(bufB, b.row(j-1));
            state = (Mp[j-1]-gap_open > Yp[j-1]-gap_ext)?0:2;
            --j;
        }
    }
    std::reverse(bufA.begin(), bufA.end());
    std::reverse(bufB.begin(), bufB.end());

    return AlignmentResult{
        std::move(bufA),
        std::move(bufB),
        static_cast<int>(bufA.size()/21),
        std::max({Mp[L2],Xp[L2],Yp[L2]})
    };
}

} // namespace sa