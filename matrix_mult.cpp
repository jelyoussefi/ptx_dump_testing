#include "matrix_ops.hpp"

namespace sycl = cl::sycl;

template <typename T>
void matrix_mult( sycl::queue q, T* inA, T* inB, T* outC, 
                  const size_t M, const size_t N, const size_t K) 
{    
    q.submit([&] (sycl::handler& cgh) {

        // Kernel submission
        cgh.parallel_for<class MatrixMul>(sycl::range<1>(M*N), [=] (sycl::id<1> idx) {
            size_t x = idx % N;
            size_t y = idx / N;

            T sum = 0;
            for (auto k=0; k<K; k++) {
                sum += inA[y*K+k] * inB[k*N+x];
            }
            outC[y*N+x] = sum;
        });
        
    }).wait();
    
}

template void 
matrix_mult<float>( sycl::queue, float*, float*, float*, const size_t, const size_t, const size_t);
