#pragma once

#ifndef REACTOR_CUDA_UTILITY_HPP_INCLUDED
#define REACTOR_CUDA_UTILITY_HPP_INCLUDED

namespace reactor
{
  namespace cuda
  {
    template <typename Synchronizable>
    inline void synchronize(Synchronizable const & _) throw(reactor::cuda::exception)
    {
      _.synchronize();
    }

    template <typename Queryable>
    inline bool query(Queryable const & _) throw(reactor::cuda::exception)
    {
      return _.query();
    }

    template <typename Notifiable, typename Event>
    inline void notify(Notifiable const & _, Event const & evt)
    {
      _.notify(evt);
    }

    template <typename Waitable, typename Event>
    inline void wait(Waitable const & _, Event const & evt)
    {
      _.wait(evt);
    }
  }
}


#endif//REACTOR_CUDA_UTILITY_HPP_INCLUDED
