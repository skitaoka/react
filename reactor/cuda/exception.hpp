#pragma once

#ifndef REACTOR_CUDA_EXCEPTION_HPP_INCLUDED
#define REACTOR_CUDA_EXCEPTION_HPP_INCLUDED

// C++ headers
#include <system_error>

// Boost headers
#ifndef BOOST_FORMAT_HPP
#include <boost/format.hpp>
#endif

// CUDA runtime headers
#ifndef __CUDA_RUNTIME_H__
#include <cuda_runtime.h>
#endif

#define REACTOR_CUDA_THROW_IF_FAILED(Statement)\
  do {\
    if (cudaError_t const e = (Statement)) {\
      throw reactor::cuda::exception(__FILE__, __LINE__, e);\
    }\
  } while (0)

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
      class category: public std::error_category
      {
      public:
        char const * name() const throw()
        {
          return "cuda exception";
        }

        std::string message(int const e) const throw()
        {
          return ::cudaGetErrorString(static_cast<cudaError_t>(e));
        }
      };

    public:
      inline exception(char const * const file, int const line, cudaError_t const e) throw()
        : std::system_error(e, reactor::cuda::exception::category(),
            boost::str(boost::format("%1%(%2%)") % file % line))
      {
      }
    };
  }
}

#endif//REACTOR_CUDA_EXCEPTION_HPP_INCLUDED
