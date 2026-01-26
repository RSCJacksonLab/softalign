cmake_minimum_required(VERSION 3.17)
project(softalign LANGUAGES CXX)

# Prefer modern FindPython to ensure we build against the active interpreter.
if(POLICY CMP0148)
  cmake_policy(SET CMP0148 NEW)
endif()
set(PYBIND11_FINDPYTHON ON)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(OpenMP REQUIRED)

pybind11_add_module(_softalign
    src/dp_kernel.cpp
    src/bindings.cpp)
target_include_directories(_softalign PRIVATE include)
target_compile_features(_softalign PRIVATE cxx_std_17)
option(SOFTALIGN_NATIVE_OPT "Enable -march=native for local builds" OFF)
target_compile_options(_softalign PRIVATE -O3 -ffast-math)
if(SOFTALIGN_NATIVE_OPT)
  target_compile_options(_softalign PRIVATE -march=native)
endif()
target_link_libraries(_softalign PRIVATE OpenMP::OpenMP_CXX)

install(TARGETS _softalign
        LIBRARY DESTINATION "${SKBUILD_PLATLIB_DIR}/softalign"   # Unix (.so / .dylib)
        RUNTIME DESTINATION "${SKBUILD_PLATLIB_DIR}/softalign")  # Windows (.pyd)
