#include <vector>
#include <cstdlib>
#include <algorithm>
#include <sys/time.h>
#include <iostream>
#include <limits>
#include <cmath>

#include "matrix_ops.hpp"

using namespace std;
using namespace sycl;

const size_t M = 1024;
const size_t K = 64;
const size_t N = 256;

//-------------------------------------------------------------------------------------------
// Local functions
//-------------------------------------------------------------------------------------------

template <typename T>
inline bool in_range(T pred, T gt) 
{
    return (gt-(1e-4) <= pred && pred <= gt+(1e-4));
}

template <typename T>
bool check_result(T* A, T* B, T* C) 
{
    for (size_t m=0; m<M; m++) {
        for (size_t n=0; n<N; n++) {
            T sum = 0.0f;
            for (size_t k=0; k<K; k++) {
                sum += A[m*K+k] * B[k*N+n];
            }
            if (in_range<T>(C[m*N+n], sum) == false) {
                printf("    [[ERR]] out[%lu,%lu] is %f, but correct vaule is %f\n", m, n, C[m*N+n], sum);
                return false;
            }
                
        }
    }

    return true;
}

//-------------------------------------------------------------------------------------------
// Public functions
//-------------------------------------------------------------------------------------------

int main() {

   
    auto q = sycl::queue(sycl::gpu_selector());

    std::cout << "\n---------------------------------------------------------------------------------" << std::endl;
    std::cout << "    Device\t: " <<q.get_device().get_info<sycl::info::device::name>()<< std::endl;
    std::cout << "    CU    \t: " <<q.get_device().get_info<sycl::info::device::max_compute_units>()<< std::endl;
    std::cout << "---------------------------------------------------------------------------------" << std::endl;

    auto A = sycl::malloc_shared<float>(M*K, q);
    auto B = sycl::malloc_shared<float>(K*N, q);
    auto C = sycl::malloc_shared<float>(M*N, q);

    for (auto i=0; i<M*K; i++) {
        A[i] = (rand()%100-50)/100.0f;
    }
    for (auto i=0; i<K*N; i++) {
        B[i] = (rand()%100-50)/100.0f;
    }


    matrix_mult<float>(q, A, B, C, M, N, K);

    bool status = check_result<float>(A, B, C);

    std::cout<<"\tstatus:\t\t" << (status ? "OK" : "KO")<<std::endl; 
    
    std::cout << "---------------------------------------------------------------------------------" << std::endl;

    

  return 0;
}

