#include "softalign.hpp"
#include <immintrin.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdint>

namespace sa {
namespace {

/*─────────────────────────────────*
 * Sum the lanes of an __m256      *
 *─────────────────────────────────*/
inline float hsum(__m256 v) {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 s  = _mm_add_ps(hi, lo);
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    return _mm_cvtss_f32(s);
}

/*─────────────────────────────────*
 * JS + BLOSUM hybrid score        *
 *─────────────────────────────────*/
float hybrid_score(const float* p,
                   const float* q,
                   const float* M,
                   float         alpha)
{
    /* BLOSUM part */
    __m256 acc = _mm256_setzero_ps();
    for (int k = 0; k < 16; k += 8) {
        __m256 pv = _mm256_loadu_ps(p + k);
        __m256 qv = _mm256_loadu_ps(q + k);
        __m256 mv = _mm256_loadu_ps(M + k*20);
        acc = _mm256_fmadd_ps(pv, _mm256_mul_ps(mv, qv), acc);
    }
    float tail = 0.f;
    for (int k = 16; k < 20; ++k)
        tail += p[k] * M[k*20 + k] * q[k];
    float blosum    = hsum(acc) + tail;
    float blos_dist = 1.f - blosum;

    /* Jensen–Shannon part */
    constexpr float EPS = 1e-8f;
    float kl1 = 0.f, kl2 = 0.f;
    for (int k = 0; k < 20; ++k) {
        float pk = p[k] + EPS;
        float qk = q[k] + EPS;
        float m  = 0.5f * (pk + qk);
        kl1 += pk * std::log(pk / m);
        kl2 += qk * std::log(qk / m);
    }
    float js = std::sqrt(std::max(0.f, 0.5f * (kl1 + kl2)));

    float d = alpha * js + (1.f - alpha) * blos_dist;
    return std::isfinite(d) ? d : 1e6f;
}

static constexpr float NEG_INF = -1e30f;

/*─────────────────────────────────*
 * Push one column (20 floats + gap flag) *
 *─────────────────────────────────*/
inline void push_col(std::vector<f32>& dst,
                     const f32*        src,
                     bool              gap = false)
{
    if (gap) {
        dst.insert(dst.end(), 20, 0.f);
        dst.push_back(1.f);
    } else {
        dst.insert(dst.end(), src, src + 20);
        dst.push_back(0.f);
    }
}

} // anonymous namespace

/*───────────────────────────────────────────────────────────────────*
 | Needleman–Wunsch with affine gaps                                 |
 *───────────────────────────────────────────────────────────────────*/
AlignmentResult
nw_affine(const ProbSeq&    a,
          const ProbSeq&    b,
          const SubstMat&   M,
          float             gap_open,
          float             gap_ext,
          float             alpha)
{
    int n = a.L, m = b.L;
    auto idx = [&](int i,int j){ return i*(m+1) + j; };

    // Full DP matrices
    std::vector<f32> Mmat((n+1)*(m+1), NEG_INF),
                     Xmat((n+1)*(m+1), NEG_INF),
                     Ymat((n+1)*(m+1), NEG_INF);
    // Pointer matrix: 0=M, 1=X, 2=Y
    std::vector<uint8_t> P((n+1)*(m+1), 0);

    // Base cases
    Mmat[idx(0,0)] = 0.f;
    for (int j = 1; j <= m; ++j) {
        Ymat[idx(0,j)] = -gap_open - (j-1)*gap_ext;
        P[idx(0,j)]    = 2;
    }
    for (int i = 1; i <= n; ++i) {
        Xmat[idx(i,0)] = -gap_open - (i-1)*gap_ext;
        P[idx(i,0)]    = 1;
    }

    // Forward DP
    for (int i = 1; i <= n; ++i) {
        for (int j = 1; j <= m; ++j) {
            float s = -hybrid_score(a.row(i-1), b.row(j-1), M.data(), alpha);

            float mM = Mmat[idx(i-1,j-1)] + s;
            float mX = Xmat[idx(i-1,j-1)] + s;
            float mY = Ymat[idx(i-1,j-1)] + s;
            float bestM = std::max({mM, mX, mY});
            Mmat[idx(i,j)] = bestM;
            P[idx(i,j)] = (bestM == mM ? 0 : (bestM == mX ? 1 : 2));

            float openX = Mmat[idx(i-1,j)] - gap_open;
            float contX = Xmat[idx(i-1,j)] - gap_ext;
            Xmat[idx(i,j)] = std::max(openX, contX);

            float openY = Mmat[idx(i,j-1)] - gap_open;
            float contY = Ymat[idx(i,j-1)] - gap_ext;
            Ymat[idx(i,j)] = std::max(openY, contY);
        }
    }

    // Pick best ending state
    float endM = Mmat[idx(n,m)],
          endX = Xmat[idx(n,m)],
          endY = Ymat[idx(n,m)];
    uint8_t state = (endM >= endX && endM >= endY)
                    ? 0
                    : (endX >= endY ? 1 : 2);

    // Traceback
    std::vector<f32> bufA, bufB;
    bufA.reserve((n+m)*21);
    bufB.reserve((n+m)*21);
    int i = n, j = m;

    while (i > 0 || j > 0) {
        if (state == 0) {
            // match/mismatch
            push_col(bufA, a.row(i-1));
            push_col(bufB, b.row(j-1));
            state = P[idx(i,j)];
            --i; --j;
        }
        else if (state == 1) {
            // gap in B
            push_col(bufA, a.row(i-1));
            push_col(bufB, nullptr, true);
            float open = Mmat[idx(i-1,j)] - gap_open;
            float cont = Xmat[idx(i-1,j)] - gap_ext;
            state = (open >= cont ? 0 : 1);
            --i;
        }
        else {
            // gap in A
            push_col(bufA, nullptr, true);
            push_col(bufB, b.row(j-1));
            float open = Mmat[idx(i,j-1)] - gap_open;
            float cont = Ymat[idx(i,j-1)] - gap_ext;
            state = (open >= cont ? 0 : 2);
            --j;
        }
    }

    std::reverse(bufA.begin(), bufA.end());
    std::reverse(bufB.begin(), bufB.end());

    int L = static_cast<int>(bufA.size() / 21);
    float score = std::max({endM, endX, endY});
    return AlignmentResult{
        std::move(bufA),
        std::move(bufB),
        L,
        score
    };
}

} // namespace sa