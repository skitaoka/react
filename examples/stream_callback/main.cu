#include <iostream>

#include <reactor/cuda/context.hpp>
#include <reactor/cuda/stream.hpp>
#include <reactor/cuda/utility.hpp>

__global__ void k1() { printf("k1\n"); }
__global__ void k2() { printf("k2\n"); }
__global__ void k3() { printf("k3\n"); }

__host__ void CUDART_CB h1(cudaStream_t stream, cudaError_t status, void *userData) { printf("h1\n"); }
__host__ void CUDART_CB h2(cudaStream_t stream, cudaError_t status, void *userData) { printf("h2\n"); }

int main()
try
{
  std::ios::sync_with_stdio(false);

  reactor::cuda::context _;
  {
    reactor::cuda::stream s1;

    k1 << <1, 1, 0, s1.get() >> >();
    s1.invoke(h1);
    k2 << <1, 1, 0, s1.get() >> >();
    s1.invoke(h2);
    k3 << <1, 1, 0, s1.get() >> >();
    reactor::cuda::synchronize(s1);
  }

  return 0;
}
catch (reactor::cuda::exception const & e) {
  std::cerr << e.what() << '\n';
}
