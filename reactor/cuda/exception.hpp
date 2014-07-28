#pragma once

#ifndef REACTOR_CUDA_EXCEPTION_HPP_INCLUDED
#define REACTOR_CUDA_EXCEPTION_HPP_INCLUDED

// C++ headers
#include <system_error>

// CUDA driver api headers
#ifndef __cuda_cuda_h__
#include <cuda.h>
#endif

// CUDA runtime api headers
#ifndef __CUDA_RUNTIME_H__
#include <cuda_runtime.h>
#endif

namespace reactor
{
  namespace cuda
  {
    //
    // cuda exception
    //
    class exception: public std::system_error
    {
    public:
      class driver_api_category: public std::error_category
      {
      public:
        char const * name() const throw()
        {
          return "CUDA driver api exception";
        }

        std::string message(int const error) const
        {
          char const * msg = nullptr;
          if (::cuGetErrorString(static_cast<CUresult>(error), &msg)) {
            msg = "CUDA: unknown error occurs!";
          }
          throw msg;
        }
      };

      class runtime_api_category: public std::error_category
      {
      public:
        char const * name() const throw()
        {
          return "CUDA runtime api exception";
        }

        std::string message(int const error) const
        {
          return ::cudaGetErrorString(static_cast<cudaError_t>(error));
        }
      };

    public:
      inline exception(CUresult const error) throw()
        : std::system_error(error, reactor::cuda::exception::driver_api_category())
      {
      }

      inline exception(cudaError_t const error) throw()
        : std::system_error(error, reactor::cuda::exception::runtime_api_category())
      {
      }
    };

    template <typename ErrorCode>
    inline void throw_if_failed(ErrorCode const e) throw(reactor::cuda::exception)
    {
      if (e) {
        throw reactor::cuda::exception(e);
      }
    }
  }
}

#endif//REACTOR_CUDA_EXCEPTION_HPP_INCLUDED
