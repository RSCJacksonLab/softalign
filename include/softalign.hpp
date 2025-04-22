#pragma once
#include <vector>

namespace sa {

using f32 = float;

struct ProbSeq {
    int        L;      // number of rows (length)
    const f32* ptr;    // pointer to first element (row‑major L×20)
    inline const f32* row(int k) const { return ptr + 20*k; }
};

struct SubstMat {
    const f32* data_;
    explicit SubstMat(const f32* p) : data_(p) {}
    inline const f32* data() const { return data_; }
};

struct AlignmentResult {
    std::vector<f32> aligned_a;   // flattened (L'×21)
    std::vector<f32> aligned_b;   // "
    int              L;          // aligned length
    float            score;      // optional
};

AlignmentResult
nw_affine(const ProbSeq& a,
          const ProbSeq& b,
          const SubstMat& M,
          float gap_open,
          float gap_ext,
          float alpha);

} // namespace sa