#pragma once

#ifndef REACTOR_CUDA_EVENT_HPP_INCLUDED
#define REACTOR_CUDA_EVENT_HPP_INCLUDED

#include <memory>

#ifndef REACTOR_CUDA_EXCEPTION_HPP
#include <reactor/cuda/exception.hpp>
#endif//REACTOR_CUDA_EXCEPTION_HP

namespace reactor
{
  namespace cuda
  {
    //
    // cuda event handle
    //
    class event: public std::shared_ptr<std::remove_pointer<cudaEvent_t>::type>
    {
      typedef std::shared_ptr<std::remove_pointer<cudaEvent_t>::type> super_type;

    public:
      // init an event with cudaEventDisableTiming
      inline __host__
        event() throw(reactor::cuda::exception)
      {
        cudaEvent_t _ = nullptr;
        reactor::cuda::throw_if_failed(::cudaEventCreateWithFlags(&_, cudaEventDisableTiming));
        super_type::reset(_, [](cudaEvent_t const _)
        {
          ::cudaEventDestroy(_);
        });
      }

    public:
      inline __host__
        void synchronize() const throw(reactor::cuda::exception)
      {
        reactor::cuda::throw_if_failed(::cudaEventSynchronize(super_type::get()));
      }

      inline __host__
        bool query() const
      {
        cudaError_t const retval = ::cudaEventQuery(super_type::get());
        if (retval == cudaErrorNotReady) {
          return false;
        }
        reactor::cuda::throw_if_failed(retval);
        return true;
      }
    };
  }
}


#endif//REACTOR_CUDA_EVENT_HPP_INCLUDED
