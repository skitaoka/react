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

#ifndef REACTOR_UTILITY_STACKTRACE_HPP_INCLUDED
#include <reactor/utility/stack_trace.hpp>
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

        std::string message(int const error) const throw()
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

        std::string message(int const error) const throw()
        {
          return ::cudaGetErrorString(static_cast<cudaError_t>(error));
        }
      };

    public:
      inline exception(CUresult const error) throw()
        : exception(error, "")
      {
      }

      inline exception(CUresult const error, std::string const & message) throw()
        : std::system_error(error, reactor::cuda::exception::driver_api_category(), message)
      {
      }

      inline exception(cudaError_t const error) throw()
        : exception(error, "")
      {
      }

      inline exception(cudaError_t const error, std::string const & message) throw()
        : std::system_error(error, reactor::cuda::exception::runtime_api_category(), message)
      {
      }
    };

    template <typename ErrorCode>
    inline void throw_if_failed(ErrorCode const e) throw(reactor::cuda::exception)
    {
      if (e) {
        throw reactor::cuda::exception(e, reactor::utility::stack_trace());
      }
    }
  }
}

#endif//REACTOR_CUDA_EXCEPTION_HPP_INCLUDED
