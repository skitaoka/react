#include <iostream>

#include <reactor/cuda/context.hpp>
#include <reactor/cuda/stream.hpp>
#include <reactor/cuda/utility.hpp>

__global__ void k1() { printf("k1\n"); }
__global__ void k2() { printf("k2\n"); }
__global__ void k3() { printf("k3\n"); }
__global__ void k4() { printf("k4\n"); }
__global__ void k5() { printf("k5\n"); }
__global__ void k6() { printf("k6\n"); }

int main()
try
{
  std::ios::sync_with_stdio(false);
  {
    reactor::cuda::context _;
    {
      reactor::cuda::stream s1;
      reactor::cuda::stream s2;

      reactor::cuda::event e1;
      reactor::cuda::event e2;

      k1<<<1,1,0,s1.get()>>>(); reactor::cuda::wait  (s1, e1); // unproductive wait() (wait() is effective against precedent notify())
      k2<<<1,1,0,s1.get()>>>(); reactor::cuda::notify(s1, e2);
      k3<<<1,1,0,s1.get()>>>(); 
                              
      k4<<<1,1,0,s2.get()>>>(); reactor::cuda::notify(s2, e1); // unproductive notify
      k5<<<1,1,0,s2.get()>>>(); reactor::cuda::wait  (s2, e2); // wait above notify(s1,e2)
      k6<<<1,1,0,s2.get()>>>();
    }
  }

  return 0;
}
catch (reactor::cuda::exception const & e)
{
  std::cerr << e.what() << '\n';
}
