#pragma once

#ifndef REACTOR_CUDA_CONTEXT_HPP_INCLUDED
#define REACTOR_CUDA_CONTEXT_HPP_INCLUDED

#ifndef REACTOR_CUDA_EXCEPTION_HPP
#include <reactor/cuda/exception.hpp>
#endif//REACTOR_CUDA_EXCEPTION_HP

namespace reactor
{
  namespace cuda
  {
    //
    // cuda device context
    //
    class context
    {
    public:
      inline __host__
        context(int const device = 0) throw(reactor::cuda::exception)
        : device_(device)
      {
        reactor::cuda::throw_if_failed(::cudaSetDevice(device));
      }

      inline __host__
        void enable() const throw(reactor::cuda::exception)
      {
        int device = -1;
        reactor::cuda::throw_if_failed(::cudaGetDevice(&device));
        if (device != device_) {
          reactor::cuda::throw_if_failed(::cudaSetDevice(device_));
        }
      }

      inline __host__
        void synchronize() const throw(reactor::cuda::exception)
      {
        enable();
        reactor::cuda::throw_if_failed(::cudaDeviceSynchronize());
      }

      inline __host__
        ~context() throw()
      {
        int device = -1;
        ::cudaGetDevice(&device);
        if (device_ != device) {
          ::cudaSetDevice(device_);
        }

        ::cudaDeviceSynchronize();
        ::cudaDeviceReset();
      }

    private:
      int const device_;
    };
  }
}

#endif//REACTOR_CUDA_CONTEXT_HPP_INCLUDED
