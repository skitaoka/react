#pragma once

#ifndef REACTOR_CUDA_EXCEPTION_HPP_INCLUDED
#define REACTOR_CUDA_EXCEPTION_HPP_INCLUDED

// C++ headers
#include <system_error>
#include <algorithm>
#include <iomanip>
#include <sstream>

#ifdef _WIN32
#include <sdkddkver.h>
#include <Windows.h>
#include <dbghelp.h>
#pragma comment(lib, "dbghelp")
#endif

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
  namespace native
  {
#ifdef _WIN32
    class address_to_symbol
    {
    public:
      inline address_to_symbol()
        : ready(false)
      {
        ::SymSetOptions(SYMOPT_DEFERRED_LOADS | SYMOPT_LOAD_LINES | SYMOPT_UNDNAME);
        if (::SymInitialize(::GetCurrentProcess(), nullptr, true)) {
          ready = true;
        }
      }

      inline ~address_to_symbol()
      {
        if (ready) {
          ::SymCleanup(::GetCurrentProcess());
          ready = false;
        }
      }

      std::string operator () (void * address) const
      {
        std::ostringstream out;
        out << "0x" << std::ios::hex << address << " ";

        HANDLE const hProcess = ::GetCurrentProcess();
        DWORD64 const addr = reinterpret_cast<DWORD64>(address);


        // モジュール名の取得
        {
          IMAGEHLP_MODULE64 module = {sizeof(IMAGEHLP_MODULE64)};
          if (ready && ::SymGetModuleInfo64(hProcess, addr, &module)) {
            out << '<' << std::setw(8) << module.ModuleName << "> ";
          } else {
            out << "<unknown  > ";
          }
        }

        // 関数名の取得
        {
          // シンボル名を格納するためのバッファを確保
          char symbol_buffer[sizeof(IMAGEHLP_SYMBOL64)+MAX_PATH];
          std::memset(symbol_buffer, 0, sizeof(symbol_buffer));

          IMAGEHLP_SYMBOL64 * psymbol = reinterpret_cast<IMAGEHLP_SYMBOL64 *>(symbol_buffer);
          psymbol->SizeOfStruct = sizeof(IMAGEHLP_SYMBOL64);
          psymbol->MaxNameLength = MAX_PATH;

          DWORD64 displacement = 0;
          if (ready && ::SymGetSymFromAddr64(hProcess, addr, &displacement, psymbol)) {
            out << psymbol->Name;
          } else {
            out << "unknown";
          }
        }

        // ファイル名と行数の取得
        {
          DWORD displacement = 0;

          IMAGEHLP_LINE64 line = {sizeof(IMAGEHLP_LINE64)};
          if (ready && ::SymGetLineFromAddr64(hProcess, addr, &displacement, &line)) {
            std::string filename(line.FileName);
            {
              auto it = filename.find_last_of('\\');
              if (it != std::string::npos) {
                filename = filename.substr(it + 1);
              }
            }
            {
              auto it = filename.find_last_of('/');
              if (it != std::string::npos) {
                filename = filename.substr(it + 1);
              }
            }
            out << '(' << filename << ':' << line.LineNumber << ')';
          } else {
            out << "(unknown:-1)";
          }
        }

        return out.str();
      }

      static address_to_symbol const & instance()
      {
        static address_to_symbol _;
        return _;
      }

    private:
      bool ready;
    };
#endif
  }

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
#ifdef _WIN32
        void * backtrace[63];  // the number of frames must be less than 63.
        std::size_t const size = ::RtlCaptureStackBackTrace(0,
          sizeof(backtrace) / sizeof(backtrace[0]), backtrace, nullptr);

        std::ostringstream out;
        for (std::size_t i = 0; i < size; ++i) {
          out << reactor::native::address_to_symbol::instance()(backtrace[i]) << '\n';
        }

        throw reactor::cuda::exception(e, out.str());
#else
        throw reactor::cuda::exception(e);
#endif
      }
    }
  }
}

#endif//REACTOR_CUDA_EXCEPTION_HPP_INCLUDED
