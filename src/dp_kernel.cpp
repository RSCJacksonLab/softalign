#include "softalign.hpp"
#include <immintrin.h>
#include <algorithm>
#include <cmath>
#include <cstdint>

namespace sa {
namespace {

/* ------------------------------------------------------------- helpers */
inline float hsum(__m256 v) {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 s  = _mm_add_ps(hi, lo);
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    return _mm_cvtss_f32(s);
}

float hybrid_score(const float* p, const float* q,
                   const float* M, float alpha)
{
    /* ---------- BLOSUM part */
    __m256 acc = _mm256_setzero_ps();
    for (int k = 0; k < 16; k += 8) {
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(p + k),
                              _mm256_mul_ps(_mm256_loadu_ps(M + k*20),
                                            _mm256_loadu_ps(q + k)),
                              acc);
    }
    float tail = 0.f;
    for (int k = 16; k < 20; ++k)
        tail += p[k] * M[k*20 + k] * q[k];
    float blos_dist = 1.f - (hsum(acc) + tail);

    /* ---------- JS part */
    constexpr float EPS = 1e-8f;
    float kl1 = 0.f, kl2 = 0.f;
    for (int k = 0; k < 20; ++k) {
        float pk = p[k] + EPS, qk = q[k] + EPS;
        float m  = 0.5f*(pk + qk);
        kl1 += pk * std::log(pk / m);
        kl2 += qk * std::log(qk / m);
    }
    float js   = std::sqrt(std::max(0.f, 0.5f*(kl1 + kl2)));
    float dist = alpha*js + (1.f-alpha)*blos_dist;
    return std::isfinite(dist) ? dist : 1e6f;
}

constexpr float NEG_INF = -1e30f;

inline void push_col(std::vector<f32>& dst,
                     const f32* src,
                     bool gap=false)
{
    if (gap) { dst.insert(dst.end(), 20, 0.f); dst.push_back(1.f); }
    else      { dst.insert(dst.end(), src, src+20); dst.push_back(0.f); }
}
/* --------------------------------------------------------------------- */
} // anon

/* ===================================================================== */
AlignmentResult
nw_affine(const ProbSeq& a, const ProbSeq& b,
          const SubstMat& M,
          float gap_open, float gap_ext, float alpha)
{
    const int L1 = a.L, L2 = b.L;

    /* rolling rows */
    std::vector<f32> Mp(L2+1, NEG_INF), Xp(L2+1, NEG_INF), Yp(L2+1, NEG_INF),
                     Mc(L2+1), Xc(L2+1), Yc(L2+1);
    Mp[0] = 0.f;
    for (int j = 1; j <= L2; ++j)
        Yp[j] = -gap_open - (j-1)*gap_ext;

    /* full matrices for gap continuation */
    std::vector<f32> MM((L1+1)*(L2+1), NEG_INF),
                     XX((L1+1)*(L2+1), NEG_INF),
                     YY((L1+1)*(L2+1), NEG_INF);

    /* store row 0 */
    for (int j = 0; j <= L2; ++j) {
        MM[j] = Mp[j];  XX[j] = Xp[j];  YY[j] = Yp[j];
    }

    /* traceback flags: 0←M, 1←X, 2←Y */
    std::vector<uint8_t> TB((L1+1)*(L2+1), 0);
    auto idx = [&](int i,int j){ return i*(L2+1)+j; };

    /* ---------------- forward DP ---------------- */
    for (int i = 1; i <= L1; ++i) {
        Mc[0]=NEG_INF;
        Xc[0]=-gap_open-(i-1)*gap_ext;
        Yc[0]=NEG_INF;

        for (int j = 1; j <= L2; ++j) {
            float s = -hybrid_score(a.row(i-1), b.row(j-1), M.data(), alpha);

            float m = Mp[j-1]+s, x = Xp[j-1]+s, y = Yp[j-1]+s;
            Mc[j]   = std::max({m,x,y});
            TB[idx(i,j)] = (Mc[j]==m)?0:(Mc[j]==x?1:2);

            Xc[j] = std::max(Mp[j]-gap_open, Xp[j]-gap_ext);
            Yc[j] = std::max(Mc[j-1]-gap_open, Yc[j-1]-gap_ext);

            MM[idx(i,j)] = Mc[j];
            XX[idx(i,j)] = Xc[j];
            YY[idx(i,j)] = Yc[j];
        }
        /* store first column of row i */
        MM[idx(i,0)] = Mc[0]; XX[idx(i,0)] = Xc[0]; YY[idx(i,0)] = Yc[0];

        std::swap(Mp,Mc); std::swap(Xp,Xc); std::swap(Yp,Yc);
    }

    /* ---------------- traceback ----------------- */
    std::vector<f32> bufA, bufB;
    bufA.reserve((L1+L2)*21); bufB.reserve((L1+L2)*21);

    int i=L1,j=L2; uint8_t state = TB[idx(i,j)];

    while (i>0 || j>0) {
        if (state==0) {
            push_col(bufA,a.row(i-1)); push_col(bufB,b.row(j-1)); --i; --j;
            state = TB[idx(i,j)];
        } else if (state==1) {
            push_col(bufA,a.row(i-1)); push_col(bufB,nullptr,true); --i;
            state = (MM[idx(i,j)]-gap_open > XX[idx(i,j)]-gap_ext)?0:1;
        } else {
            push_col(bufA,nullptr,true); push_col(bufB,b.row(j-1)); --j;
            state = (MM[idx(i,j)]-gap_open > YY[idx(i,j)]-gap_ext)?0:2;
        }
    }
    std::reverse(bufA.begin(),bufA.end());
    std::reverse(bufB.begin(),bufB.end());

    return AlignmentResult{std::move(bufA), std::move(bufB),
                           static_cast<int>(bufA.size()/21),
                           std::max({Mp[L2],Xp[L2],Yp[L2]})};
}
/* ===================================================================== */
} // namespace sa