#pragma once

#include <CL/sycl.hpp>

template <typename T>
void matrix_mult( sycl::queue queue, T* inA, T* inB, T*  outC, 
                  const size_t M, const size_t N, const size_t K);