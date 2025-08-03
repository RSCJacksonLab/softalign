// In softalign/src/bindings.cpp

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h> 
#include "softalign.hpp"
#include <vector> // Required for std::copy

namespace py = pybind11;
using sa::f32; using sa::ProbSeq; using sa::SubstMat; using sa::AlignmentResult;

ProbSeq as_probseq(const py::array_t<f32>& arr)
{
    if (arr.ndim()!=2 || arr.shape(1)!=20)
        throw std::runtime_error("prob seq must be (L,20) float32");
    return {static_cast<int>(arr.shape(0)), arr.data()};
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
              
              // Create a new, empty NumPy array of the correct flat size.
              py::array_t<f32> py_aligned_a(r.aligned_a.size());
              // Get a direct pointer to the new NumPy array's memory buffer.
              f32* ptr_a = static_cast<f32*>(py_aligned_a.request().ptr);
              // Copy the data from the C++ vector into the NumPy buffer.
              std::copy(r.aligned_a.begin(), r.aligned_a.end(), ptr_a);
              // Reshape the new NumPy array to its final 2D shape.
              py_aligned_a.resize({r.L, 21});

              // Repeat the same safe process for the second sequence.
              py::array_t<f32> py_aligned_b(r.aligned_b.size());
              f32* ptr_b = static_cast<f32*>(py_aligned_b.request().ptr);
              std::copy(r.aligned_b.begin(), r.aligned_b.end(), ptr_b);
              py_aligned_b.resize({r.L, 21});

              return py::make_tuple(py_aligned_a, py_aligned_b, r.score);
          },
          py::arg("seq1"), py::arg("seq2"), py::arg("subst"),
          py::arg("gap_open")=10.f, py::arg("gap_ext")=0.5f,
          py::arg("alpha")=0.5f);
}
