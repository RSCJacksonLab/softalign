#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "softalign.hpp"

namespace py = pybind11;
using sa::f32; using sa::ProbSeq; using sa::SubstMat; using sa::AlignmentResult;

ProbSeq as_probseq(const py::array_t<f32>& arr)
{
    if (arr.ndim()!=2 || arr.shape(1)!=20)
        throw std::runtime_error("prob seq must be (L,20) float32");
    return {static_cast<int>(arr.shape(0)), arr.data()};
}

py::array_t<f32> make_2d(const std::vector<f32>& v, int L)
{
    return py::array_t<f32>({L,21},
                            {21*sizeof(f32), sizeof(f32)},
                            v.data(),                  // no copyso
                            py::none());               // base = none
}

PYBIND11_MODULE(_softalign, m) {
    m.def("nw_affine",
          [](py::array_t<f32, py::array::c_style|py::array::forcecast> a,
             py::array_t<f32, py::array::c_style|py::array::forcecast> b,
             py::array_t<f32, py::array::c_style|py::array::forcecast> subst,
             float gap_open, float gap_ext, float alpha)
          {
              ProbSeq A = as_probseq(a);
              ProbSeq B = as_probseq(b);
              SubstMat M(subst.data());
              AlignmentResult r = sa::nw_affine(A,B,M,gap_open,gap_ext,alpha);
              return py::make_tuple(make_2d(r.aligned_a, r.L),
                                    make_2d(r.aligned_b, r.L),
                                    r.score);
          },
          py::arg("seq1"), py::arg("seq2"), py::arg("subst"),
          py::arg("gap_open")=10.f, py::arg("gap_ext")=0.5f,
          py::arg("alpha")=0.5f);
}