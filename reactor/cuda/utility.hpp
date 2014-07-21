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
  }
}


#endif//REACTOR_CUDA_UTILITY_HPP_INCLUDED
