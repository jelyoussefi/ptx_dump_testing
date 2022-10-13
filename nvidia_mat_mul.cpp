#include <stdio.h>
#include <CL/sycl.hpp>


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


//-------------------------------------------------------------------------------------------
// Public functions
//-------------------------------------------------------------------------------------------

int main() {

   
    auto q = sycl::queue(sycl::gpu_selector());


    printf("Device\t: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
   
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

    printf("\tstatus:\t%s\n", (status ? "OK" : "KO"));

  return 0;
}

