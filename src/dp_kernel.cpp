#include "softalign.hpp"
#include <immintrin.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdint>

namespace sa {
namespace {

// Enum to make pointer matrix states clearer
enum class Pointer : uint8_t {
    STOP = 0,
    MATCH_FROM_M,
    MATCH_FROM_X,
    MATCH_FROM_Y,
    GAP_IN_B, // Corresponds to X matrix
    GAP_IN_A  // Corresponds to Y matrix
};

inline float hsum(__m256 v) {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 s  = _mm_add_ps(hi, lo);
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    return _mm_cvtss_f32(s);
}

// JS + BLOSUM hybrid score
float hybrid_score(const float* p,
                   const float* q,
                   const float* M,
                   float         alpha)
{
    // BLOSUM part
    __m256 acc = _mm256_setzero_ps();
    for (int k = 0; k < 16; k += 8) {
        __m256 pv = _mm256_loadu_ps(p + k);
        __m256 qv = _mm256_loadu_ps(q + k);
        __m256 mv = _mm256_loadu_ps(M + k*20); // This assumes M is row-major for p and col-major for q
        acc = _mm256_fmadd_ps(pv, _mm256_mul_ps(mv, qv), acc);
    }
    float tail = 0.f;
    for (int k = 16; k < 20; ++k)
        tail += p[k] * M[k*20 + k] * q[k];
    float blosum    = hsum(acc) + tail;
    float blos_dist = 1.f - blosum;

    // Jensen–Shannon part
    constexpr float EPS = 1e-8f;
    float kl1 = 0.f, kl2 = 0.f;
    for (int k = 0; k < 20; ++k) {
        float pk = p[k] + EPS;
        float qk = q[k] + EPS;
        float m  = 0.5f * (pk + qk);
        if (pk > 0) kl1 += pk * std::log(pk / m);
        if (qk > 0) kl2 += qk * std::log(qk / m);
    }
    float js = std::sqrt(std::max(0.f, 0.5f * (kl1 + kl2)));

    float d = alpha * js + (1.f - alpha) * blos_dist;
    return std::isfinite(d) ? d : 1e6f;
}

static constexpr float NEG_INF = -1e30f;

inline void push_col(std::vector<f32>& dst,
                     const f32* src,
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

// Needleman–Wunsch with affine gaps
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

    std::vector<f32> Mmat((n+1)*(m+1), NEG_INF),
                     Xmat((n+1)*(m+1), NEG_INF),
                     Ymat((n+1)*(m+1), NEG_INF);
    std::vector<Pointer> Ptr_M((n+1)*(m+1)),
                         Ptr_X((n+1)*(m+1)),
                         Ptr_Y((n+1)*(m+1));

    Mmat[idx(0,0)] = 0.f;

    for (int i = 1; i <= n; ++i) {
        Xmat[idx(i,0)] = -gap_open - (i-1)*gap_ext;
        Ptr_X[idx(i,0)] = Pointer::GAP_IN_B;
    }
    for (int j = 1; j <= m; ++j) {
        Ymat[idx(0,j)] = -gap_open - (j-1)*gap_ext;
        Ptr_Y[idx(0,j)] = Pointer::GAP_IN_A;
    }

    // Forward DP
    for (int i = 1; i <= n; ++i) {
        for (int j = 1; j <= m; ++j) {
            float s = -hybrid_score(a.row(i-1), b.row(j-1), M.data(), alpha);

            // M matrix
            float m_from_m = Mmat[idx(i-1,j-1)] + s;
            float m_from_x = Xmat[idx(i-1,j-1)] + s;
            float m_from_y = Ymat[idx(i-1,j-1)] + s;
            if (m_from_m >= m_from_x && m_from_m >= m_from_y) {
                Mmat[idx(i,j)] = m_from_m;
                Ptr_M[idx(i,j)] = Pointer::MATCH_FROM_M;
            } else if (m_from_x >= m_from_y) {
                Mmat[idx(i,j)] = m_from_x;
                Ptr_M[idx(i,j)] = Pointer::MATCH_FROM_X;
            } else {
                Mmat[idx(i,j)] = m_from_y;
                Ptr_M[idx(i,j)] = Pointer::MATCH_FROM_Y;
            }

            // X matrix (gap in B)
            float x_open = Mmat[idx(i-1,j)] - gap_open;
            float x_ext = Xmat[idx(i-1,j)] - gap_ext;
            if (x_open >= x_ext) {
                Xmat[idx(i,j)] = x_open;
                Ptr_X[idx(i,j)] = Pointer::MATCH_FROM_M; // came from M
            } else {
                Xmat[idx(i,j)] = x_ext;
                Ptr_X[idx(i,j)] = Pointer::GAP_IN_B; // came from X
            }

            // Y matrix (gap in A)
            float y_open = Mmat[idx(i,j-1)] - gap_open;
            float y_ext = Ymat[idx(i,j-1)] - gap_ext;
            if (y_open >= y_ext) {
                Ymat[idx(i,j)] = y_open;
                Ptr_Y[idx(i,j)] = Pointer::MATCH_FROM_M; // came from M
            } else {
                Ymat[idx(i,j)] = y_ext;
                Ptr_Y[idx(i,j)] = Pointer::GAP_IN_A; // came from Y
            }
        }
    }

    // Traceback
    std::vector<f32> bufA, bufB;
    bufA.reserve((n+m)*21);
    bufB.reserve((n+m)*21);
    int i = n, j = m;

    float score_m = Mmat[idx(n,m)];
    float score_x = Xmat[idx(n,m)];
    float score_y = Ymat[idx(n,m)];
    
    enum class Matrix { M, X, Y };
    Matrix current_matrix;
    if (score_m >= score_x && score_m >= score_y) {
        current_matrix = Matrix::M;
    } else if (score_x >= score_y) {
        current_matrix = Matrix::X;
    } else {
        current_matrix = Matrix::Y;
    }
    
    // --- Start of MINIMAL FIX ---

    // Main traceback loop runs only while both sequences have characters left.
    while (i > 0 && j > 0) {
        if (current_matrix == Matrix::M) {
            push_col(bufA, a.row(i-1));
            push_col(bufB, b.row(j-1));
            Pointer ptr = Ptr_M[idx(i,j)];
            i--; j--;
            if (ptr == Pointer::MATCH_FROM_M) current_matrix = Matrix::M;
            else if (ptr == Pointer::MATCH_FROM_X) current_matrix = Matrix::X;
            else current_matrix = Matrix::Y; // MATCH_FROM_Y
        } else if (current_matrix == Matrix::X) {
            push_col(bufA, a.row(i-1));
            push_col(bufB, nullptr, true);
            Pointer ptr = Ptr_X[idx(i,j)];
            i--;
            if (ptr == Pointer::MATCH_FROM_M) current_matrix = Matrix::M;
            else current_matrix = Matrix::X; // Corresponds to GAP_IN_B
        } else { // Matrix::Y
            push_col(bufA, nullptr, true);
            push_col(bufB, b.row(j-1));
            Pointer ptr = Ptr_Y[idx(i,j)];
            j--;
            if (ptr == Pointer::MATCH_FROM_M) current_matrix = Matrix::M;
            else current_matrix = Matrix::Y; // Corresponds to GAP_IN_A
        }
    }

    // After the main loop, handle any remaining characters (the alignment tails).
    
    // If sequence 'a' has remaining characters, align them with gaps in 'b'.
    while (i > 0) {
        push_col(bufA, a.row(i-1));
        push_col(bufB, nullptr, true);
        i--;
    }

    // If sequence 'b' has remaining characters, align them with gaps in 'a'.
    while (j > 0) {
        push_col(bufA, nullptr, true);
        push_col(bufB, b.row(j-1));
        j--;
    }
    
    // --- End of MINIMAL FIX ---

    std::reverse(bufA.begin(), bufA.end());
    std::reverse(bufB.begin(), bufB.end());

    int L = static_cast<int>(bufA.size() / 21);
    float score = std::max({score_m, score_x, score_y});
    return AlignmentResult{
        std::move(bufA),
        std::move(bufB),
        L,
        score
    };
}

} // namespace sa
