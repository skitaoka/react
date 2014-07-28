#pragma once

#ifndef REACTOR_CUDA_STREAM_HPP_INCLUDED
#define REACTOR_CUDA_STREAM_HPP_INCLUDED

#include <memory>

#ifndef REACTOR_CUDA_EVENT_HPP_INCLUDED
#include <reactor/cuda/event.hpp>
#endif//REACTOR_CUDA_EVENT_HPP_INCLUDED

namespace reactor
{
  namespace cuda
  {
    //
    // cuda stream handle
    //
    class stream: public std::shared_ptr<std::remove_pointer<cudaStream_t>::type>
    {
      typedef std::shared_ptr<std::remove_pointer<cudaStream_t>::type> super_type;

    public:
      // init stream with cudaStreamNonBlocking
      inline __host__
        stream() throw(reactor::cuda::exception)
      {
        cudaStream_t _ = nullptr;
        reactor::cuda::throw_if_failed(::cudaStreamCreateWithFlags(&_, cudaStreamNonBlocking));
        super_type::reset(_, [](cudaStream_t const _)
        {
          ::cudaStreamSynchronize(_);
          ::cudaStreamDestroy(_);
        });
      }

    public:
      inline __host__
        void synchronize() const throw(reactor::cuda::exception)
      {
          reactor::cuda::throw_if_failed(::cudaStreamSynchronize(super_type::get()));
      }

      inline __host__
        bool query() const
      {
        cudaError_t const retval = ::cudaStreamQuery(super_type::get());
        if (retval == cudaErrorNotReady) {
          return false;
        }
        reactor::cuda::throw_if_failed(retval);
        return true;
      }

    public:
      inline __host__
        void notify(reactor::cuda::event const & evt) const throw(reactor::cuda::exception)
      {
        reactor::cuda::throw_if_failed(::cudaEventRecord(evt.get(), super_type::get()));
      }

      inline __host__
        void wait(reactor::cuda::event const & evt) const throw(reactor::cuda::exception)
      {
        reactor::cuda::throw_if_failed(::cudaStreamWaitEvent(super_type::get(), evt.get(), 0));
      }

    public:
      inline __host__
        void invoke(cudaStreamCallback_t callback) const throw(reactor::cuda::exception)
      {
        this->invoke<void>(callback, nullptr);
      }

      template <typename T>
      inline __host__
        void invoke(cudaStreamCallback_t callback, T * const data) const throw(reactor::cuda::exception)
      {
        reactor::cuda::throw_if_failed(::cudaStreamAddCallback(super_type::get(), callback, data, 0));
      }

    public:
      inline __host__
        void attach(void * const ptr, bool const enable_sync = false) const throw(reactor::cuda::exception)
      {
        reactor::cuda::throw_if_failed(::cudaStreamAttachMemAsync(super_type::get(), ptr, 0, cudaMemAttachGlobal));
        if (enable_sync) {
          this->synchronize();
        }
      }

      inline __host__
        void detach(void * const ptr, bool const enable_sync = false) const throw(reactor::cuda::exception)
      {
        reactor::cuda::throw_if_failed(::cudaStreamAttachMemAsync(super_type::get(), ptr, 0, cudaMemAttachHost));
        if (enable_sync) {
          this->synchronize();
        }
      }
    };
  }
}

#endif//REACTOR_CUDA_STREAM_HPP_INCLUDED
