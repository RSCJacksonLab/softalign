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
include(CheckIncludeFileCXX)

# Provide a clearer failure mode for source builds on macOS without a toolchain.
check_include_file_cxx(vector SOFTALIGN_HAVE_VECTOR)
set(SOFTALIGN_LIBCXX_INCLUDE_DIR "")
set(SOFTALIGN_LIBCXX_LIB_DIR "")
if(NOT SOFTALIGN_HAVE_VECTOR AND APPLE)
  find_path(
    SOFTALIGN_LIBCXX_INCLUDE_DIR
    vector
    PATHS
      /opt/homebrew/opt/llvm/include/c++/v1
      /usr/local/opt/llvm/include/c++/v1
    NO_DEFAULT_PATH
  )
  if(SOFTALIGN_LIBCXX_INCLUDE_DIR)
    message(STATUS "Using Homebrew libc++ headers: ${SOFTALIGN_LIBCXX_INCLUDE_DIR}")
    # Derive the Homebrew LLVM prefix and its lib directory.
    get_filename_component(_cxx_dir "${SOFTALIGN_LIBCXX_INCLUDE_DIR}" DIRECTORY)
    get_filename_component(_include_dir "${_cxx_dir}" DIRECTORY)
    get_filename_component(_llvm_prefix "${_include_dir}" DIRECTORY)
    set(SOFTALIGN_LIBCXX_LIB_DIR "${_llvm_prefix}/lib")
  endif()
endif()

pybind11_add_module(_softalign
    src/dp_kernel.cpp
    src/bindings.cpp)
target_include_directories(_softalign PRIVATE include)
if(SOFTALIGN_LIBCXX_INCLUDE_DIR)
  target_include_directories(_softalign SYSTEM PRIVATE "${SOFTALIGN_LIBCXX_INCLUDE_DIR}")
  target_compile_options(_softalign PRIVATE -stdlib=libc++)
  target_link_options(_softalign PRIVATE -stdlib=libc++)
  if(SOFTALIGN_LIBCXX_LIB_DIR)
    # When using Homebrew libc++ headers, also link to the matching runtime.
    find_library(SOFTALIGN_LIBCXX_LIB NAMES c++ libc++ HINTS "${SOFTALIGN_LIBCXX_LIB_DIR}")
    find_library(SOFTALIGN_LIBCXXABI_LIB NAMES c++abi libc++abi HINTS "${SOFTALIGN_LIBCXX_LIB_DIR}")
    if(SOFTALIGN_LIBCXX_LIB)
      target_link_libraries(_softalign PRIVATE "${SOFTALIGN_LIBCXX_LIB}")
      target_link_options(_softalign PRIVATE "-Wl,-rpath,${SOFTALIGN_LIBCXX_LIB_DIR}")
      message(STATUS "Linking against Homebrew libc++: ${SOFTALIGN_LIBCXX_LIB_DIR}")
    endif()
    if(SOFTALIGN_LIBCXXABI_LIB)
      target_link_libraries(_softalign PRIVATE "${SOFTALIGN_LIBCXXABI_LIB}")
    endif()
  endif()
endif()
target_compile_features(_softalign PRIVATE cxx_std_17)
option(SOFTALIGN_NATIVE_OPT "Enable -march=native for local builds" OFF)
target_compile_options(_softalign PRIVATE -O3 -ffast-math)
if(SOFTALIGN_NATIVE_OPT)
  target_compile_options(_softalign PRIVATE -march=native)
endif()
target_link_libraries(_softalign PRIVATE OpenMP::OpenMP_CXX)

# Fail fast with actionable guidance if the standard library headers are absent.
if(NOT SOFTALIGN_HAVE_VECTOR AND NOT SOFTALIGN_LIBCXX_INCLUDE_DIR)
  message(FATAL_ERROR
    "C++ standard library headers were not found. "
    "For end users, install a prebuilt wheel (pip install softalign). "
    "Building from source on macOS requires Xcode Command Line Tools "
    "or a Homebrew LLVM toolchain."
  )
endif()

install(TARGETS _softalign
        LIBRARY DESTINATION "${SKBUILD_PLATLIB_DIR}/softalign"   # Unix (.so / .dylib)
        RUNTIME DESTINATION "${SKBUILD_PLATLIB_DIR}/softalign")  # Windows (.pyd)
