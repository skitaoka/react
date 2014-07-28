#pragma once

#ifndef REACTOR_CUDA_MEMORY_HPP_INCLUDED
#define REACTOR_CUDA_MEMORY_HPP_INCLUDED

// C++ headers
#include <memory>

// Reactor headers
#ifndef REACTOR_CUDA_EXCEPTION_HPP_INCLUDED
#include <reactor/cuda/exception.hpp>
#endif//REACTOR_CUDA_EXCEPTION_HPP_INCLUDED

namespace reactor
{
  namespace cuda
  {
    //
    // cuda memory allocator
    //
    template <typename T>
    inline __host__
      std::shared_ptr<T> malloc(std::size_t const size, bool const is_managed = false) throw(reactor::cuda::exception)
    {
      void * _ = nullptr;
      if (is_managed) {
        reactor::cuda::throw_if_failed(::cudaMallocManaged(&_, size * sizeof(T), cudaMemAttachGlobal));
      } else {
        reactor::cuda::throw_if_failed(::cudaMalloc(&_, size * sizeof(T)));
      }
      return std::shared_ptr<T>(reinterpret_cast<T*>(_), [](T * const _) { ::cudaFree(_); });
    }

    //
    // cuda memory copy
    //
    template <typename T>
    inline __host__
      void memcpy(T * const dst, T const * const src, std::size_t const size, cudaStream_t stream) throw(reactor::cuda::exception)
    {
      reactor::cuda::throw_if_failed(::cudaMemcpyAsync(dst, src, size * sizeof(T), cudaMemcpyDefault, stream));
    }
  }
}

#endif//REACTOR_CUDA_MEMORY_HPP_INCLUDED
