//!
//! <reactor/math/blas.hpp> is a wrapper of <nvblas.h>
//!
#pragma once

#ifndef REACTOR_MATH_BLAS_H_INCLUDED
#define REACTOR_MATH_BLAS_H_INCLUDED

#include <nvblas.h>

#pragma comment(lib, "nvblas")

namespace reactor
{
  namespace math
  {
    //!
    //! C <= alpha A B + beta C
    //!
    template <typename T>
    inline void gemm(
      char const   transa,
      char const   transb,
      int  const   m     ,
      int  const   n     ,
      int  const   k     ,
      T    const   alpha ,
      T    const * a     , int const lda,
      T    const * b     , int const ldb,
      T    const   beta  ,
      T          * c     , int const ldc);

    template <>
    inline void gemm(
      char  const   transa,
      char  const   transb,
      int   const   m     ,
      int   const   n     ,
      int   const   k     ,
      float const   alpha ,
      float const * a     , int const lda,
      float const * b     , int const ldb,
      float const   beta  ,
      float       * c     , int const ldc)
    {
      sgemm(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
    }

    template <>
    void gemm(
      char   const   transa,
      char   const   transb,
      int    const   m     ,
      int    const   n     ,
      int    const   k     ,
      double const   alpha ,
      double const * a     , int const lda,
      double const * b     , int const ldb,
      double const   beta  ,
      double       * c     , int const ldc)
    {
      dgemm(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
    }

    template <>
    inline void gemm(
      char      const   transa,
      char      const   transb,
      int       const   m     ,
      int       const   n     ,
      int       const   k     ,
      cuComplex const   alpha ,
      cuComplex const * a     , int const lda,
      cuComplex const * b     , int const ldb,
      cuComplex const   beta  ,
      cuComplex       * c     , int const ldc)
    {
      cgemm(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
    }

    template <>
    inline void gemm(
      char            const   transa,
      char            const   transb,
      int             const   m     ,
      int             const   n     ,
      int             const   k     ,
      cuDoubleComplex const   alpha ,
      cuDoubleComplex const * a     , int const lda,
      cuDoubleComplex const * b     , int const ldb,
      cuDoubleComplex const   beta  ,
      cuDoubleComplex       * c     , int const ldc)
    {
      zgemm(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
    }
  }
}

namespace reactor
{
  namespace math
  {
    //!
    //! C <= alpha A A' + beta * C
    //!
    template <typename T>
    inline void syrk(
      char const   uplo ,
      char const   trans,
      int  const   n    ,
      int  const   k    ,
      T    const   alpha,
      T    const * a    , int const lda,
      T    const   beta ,
      T          * c    , int const ldc);

    template <>
    inline void syrk(
      char  const   uplo ,
      char  const   trans,
      int   const   n    ,
      int   const   k    ,
      float const   alpha,
      float const * a    , int const lda,
      float const   beta ,
      float       * c    , int const ldc)
    {
      ssyrk(&uplo, &trans, &n, &k, &alpha, a, &lda, &beta, c, &ldc);
    }

    template <>
    inline void syrk(
      char   const   uplo ,
      char   const   trans,
      int    const   n    ,
      int    const   k    ,
      double const   alpha,
      double const * a    , int const lda,
      double const   beta ,
      double       * c    , int const ldc)
    {
      dsyrk(&uplo, &trans, &n, &k, &alpha, a, &lda, &beta, c, &ldc);
    }

    template <>
    inline void syrk(
      char      const   uplo ,
      char      const   trans,
      int       const   n    ,
      int       const   k    ,
      cuComplex const   alpha,
      cuComplex const * a    , int const lda,
      cuComplex const   beta ,
      cuComplex       * c    , int const ldc)
    {
      csyrk(&uplo, &trans, &n, &k, &alpha, a, &lda, &beta, c, &ldc);
    }

    template <>
    inline void syrk(
      char            const   uplo ,
      char            const   trans,
      int             const   n    ,
      int             const   k    ,
      cuDoubleComplex const   alpha,
      cuDoubleComplex const * a    , int const lda,
      cuDoubleComplex const   beta ,
      cuDoubleComplex       * c    , int const ldc)
    {
      zsyrk(&uplo, &trans, &n, &k, &alpha, a, &lda, &beta, c, &ldc);
    }
  }
}

namespace reactor
{
  namespace math
  {
    //!
    //! solve A X = alpha B
    //!
    template <typename T>
    inline void trsm(
      char const   side  ,
      char const   uplo  ,
      char const   transa,
      char const   diag  ,
      int  const   m     ,
      int  const   n     ,
      T    const   alpha ,
      T    const * a     , int const lda,
      T          * b     , int const ldb);

    template <>
    inline void trsm(
      char  const   side  ,
      char  const   uplo  ,
      char  const   transa,
      char  const   diag  ,
      int   const   m     ,
      int   const   n     ,
      float const   alpha ,
      float const * a     , int const lda,
      float       * b     , int const ldb)
    {
      strsm(&side, &uplo, &transa, &diag, &m, &n, &alpha, a, &lda, b, &ldb);
    }

    template <>
    inline void trsm(
      char   const   side  ,
      char   const   uplo  ,
      char   const   transa,
      char   const   diag  ,
      int    const   m     ,
      int    const   n     ,
      double const   alpha ,
      double const * a     , int const lda,
      double       * b     , int const ldb)
    {
      dtrsm(&side, &uplo, &transa, &diag, &m, &n, &alpha, a, &lda, b, &ldb);
    }

    template <>
    inline void trsm(
      char      const   side  ,
      char      const   uplo  ,
      char      const   transa,
      char      const   diag  ,
      int       const   m     ,
      int       const   n     ,
      cuComplex const   alpha ,
      cuComplex const * a     , int const lda,
      cuComplex       * b     , int const ldb)
    {
      ctrsm(&side, &uplo, &transa, &diag, &m, &n, &alpha, a, &lda, b, &ldb);
    }

    template <>
    inline void trsm(
      char            const   side  ,
      char            const   uplo  ,
      char            const   transa,
      char            const   diag  ,
      int             const   m     ,
      int             const   n     ,
      cuDoubleComplex const   alpha ,
      cuDoubleComplex const * a     , int const lda,
      cuDoubleComplex       * b     , int const ldb)
    {
      ztrsm(&side, &uplo, &transa, &diag, &m, &n, &alpha, a, &lda, b, &ldb);
    }
  }
}

namespace reactor
{
  namespace math
  {
    //!
    //! C <= alpha A B + beta C, or
    //! C <= alpha B A + beta C.
    //!
    template <typename T>
    inline void symm(
      char const   side ,
      char const   uplo ,
      int  const   m    ,
      int  const   n    ,
      T    const   alpha,
      T    const * a    , int const lda,
      T    const * b    , int const ldb,
      T    const   beta ,
      T          * c    , int const ldc);

    template <>
    inline void symm(
      char  const   side ,
      char  const   uplo ,
      int   const   m    ,
      int   const   n    ,
      float const   alpha,
      float const * a    , int const lda,
      float const * b    , int const ldb,
      float const   beta ,
      float       * c    , int const ldc)
    {
      ssymm(&side, &uplo, &m, &n, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
    }

    template <>
    inline void symm(
      char   const   side ,
      char   const   uplo ,
      int    const   m    ,
      int    const   n    ,
      double const   alpha,
      double const * a    , int const lda,
      double const * b    , int const ldb,
      double const   beta ,
      double       * c    , int const ldc)
    {
      dsymm(&side, &uplo, &m, &n, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
    }

    template <>
    inline void symm(
      char      const   side ,
      char      const   uplo ,
      int       const   m    ,
      int       const   n    ,
      cuComplex const   alpha,
      cuComplex const * a    , int const lda,
      cuComplex const * b    , int const ldb,
      cuComplex const   beta ,
      cuComplex       * c    , int const ldc)
    {
      csymm(&side, &uplo, &m, &n, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
    }

    template <>
    inline void symm(
      char            const   side ,
      char            const   uplo ,
      int             const   m    ,
      int             const   n    ,
      cuDoubleComplex const   alpha,
      cuDoubleComplex const * a    , int const lda,
      cuDoubleComplex const * b    , int const ldb,
      cuDoubleComplex const   beta ,
      cuDoubleComplex       * c    , int const ldc)
    {
      zsymm(&side, &uplo, &m, &n, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
    }
  }
}

namespace reactor
{
  namespace math
  {
    //!
    //! C <= alpha A  B' + alpha B  A' + beta C, or
    //! C <= alpha A' B  + alpha B' A  + beta C.
    //!
    template <typename T>
    inline void syr2k(
      char const   uplo ,
      char const   trans,
      int  const   n    ,
      int  const   k    ,
      T    const   alpha,
      T    const * a    , int const lda,
      T    const * b    , int const ldb,
      T    const   beta ,
      T          * c    , int const ldc);

    template <typename T>
    inline void syr2k(
      char  const   uplo ,
      char  const   trans,
      int   const   n    ,
      int   const   k    ,
      float const   alpha,
      float const * a    , int const lda,
      float const * b    , int const ldb,
      float const   beta ,
      float       * c    , int const ldc)
    {
      ssyr2k(&uplo, &trans, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
    }

    template <typename T>
    inline void syr2k(
      char   const   uplo ,
      char   const   trans,
      int    const   n    ,
      int    const   k    ,
      double const   alpha,
      double const * a    , int const lda,
      double const * b    , int const ldb,
      double const   beta ,
      double       * c    , int const ldc)
    {
      dsyr2k(&uplo, &trans, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
    }

    template <typename T>
    inline void syr2k(
      char      const   uplo ,
      char      const   trans,
      int       const   n    ,
      int       const   k    ,
      cuComplex const   alpha,
      cuComplex const * a    , int const lda,
      cuComplex const * b    , int const ldb,
      cuComplex const   beta ,
      cuComplex       * c    , int const ldc)
    {
      csyr2k(&uplo, &trans, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
    }

    template <typename T>
    inline void syr2k(
      char            const   uplo ,
      char            const   trans,
      int             const   n    ,
      int             const   k    ,
      cuDoubleComplex const   alpha,
      cuDoubleComplex const * a    , int const lda,
      cuDoubleComplex const * b    , int const ldb,
      cuDoubleComplex const   beta ,
      cuDoubleComplex       * c    , int const ldc)
    {
      zsyr2k(&uplo, &trans, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
    }
  }
}

namespace reactor
{
  namespace math
  {
    //!
    //! B <= alpha A B, or
    //! B <= alpha B A.
    //!
    template <typename T>
    inline void trmm(
      char const   side  ,
      char const   uplo  ,
      char const   transa,
      char const   diag  ,
      int  const   m     ,
      int  const   n     ,
      T    const   alpha ,
      T    const * a     , int const lda,
      T          * b     , int const ldb);

    template <>
    inline void trmm(
      char  const   side  ,
      char  const   uplo  ,
      char  const   transa,
      char  const   diag  ,
      int   const   m     ,
      int   const   n     ,
      float const   alpha ,
      float const * a     , int const lda,
      float       * b     , int const ldb)
    {
      strmm(&side, &uplo, &transa, &diag, &m, &n, &alpha, a, &lda, b, &ldb);
    }

    template <>
    inline void trmm(
      char   const   side  ,
      char   const   uplo  ,
      char   const   transa,
      char   const   diag  ,
      int    const   m     ,
      int    const   n     ,
      double const   alpha ,
      double const * a     , int const lda,
      double       * b     , int const ldb)
    {
      dtrmm(&side, &uplo, &transa, &diag, &m, &n, &alpha, a, &lda, b, &ldb);
    }

    template <>
    inline void trmm(
      char      const   side  ,
      char      const   uplo  ,
      char      const   transa,
      char      const   diag  ,
      int       const   m     ,
      int       const   n     ,
      cuComplex const   alpha ,
      cuComplex const * a     , int const lda,
      cuComplex       * b     , int const ldb)
    {
      ctrmm(&side, &uplo, &transa, &diag, &m, &n, &alpha, a, &lda, b, &ldb);
    }

    template <>
    inline void trmm(
      char            const   side  ,
      char            const   uplo  ,
      char            const   transa,
      char            const   diag  ,
      int             const   m     ,
      int             const   n     ,
      cuDoubleComplex const   alpha ,
      cuDoubleComplex const * a     , int const lda,
      cuDoubleComplex       * b     , int const ldb)
    {
      ztrmm(&side, &uplo, &transa, &diag, &m, &n, &alpha, a, &lda, b, &ldb);
    }
  }
}

#endif//REACTOR_MATH_BLAS_H_INCLUDED
