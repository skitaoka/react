#pragma once

#ifndef REACTOR_UTILITY_STACKTRACE_HPP_INCLUDED
#define REACTOR_UTILITY_STACKTRACE_HPP_INCLUDED

// C++ headers
#include <ostream>
#include <iomanip>
#include <sstream>

#ifdef _WIN32
#include <sdkddkver.h>
#include <Windows.h>
#include <dbghelp.h>
#pragma comment(lib, "dbghelp")
#endif

#ifdef __GNUC__
#include <memory>
#include <vector>

// glibc の機能を使う
// env needs "yum install binutils-devel"
// g++ needs options: -g
// ld  needs options: -lbfd -liberty -ldl
#include <bfd.h> 
#include <dlfcn.h>
#include <execinfo.h>
#include <cxxabi.h>
#endif


namespace reactor
{
  namespace utility
  {
    std::string path_to_filename(std::string path)
    {
      {
        auto it = path.find_last_of('\\');
        if (it != std::string::npos) {
          path = path.substr(it + 1);
        }
      }
      {
        auto it = path.find_last_of('/');
        if (it != std::string::npos) {
          path = path.substr(it + 1);
        }
      }
      return path;
    }

#if defined(_WIN32) || defined(__GNUC__)
    class address_to_symbol
    {
    private:
      inline address_to_symbol()
        : ready_(false)
#ifdef __GNUC__
        , bfd_(bfd_openr("/proc/self/exe", NULL), bfd_close)
        , section_(NULL)
#endif
      {
#ifdef _WIN32
        ::SymSetOptions(SYMOPT_DEFERRED_LOADS | SYMOPT_LOAD_LINES | SYMOPT_UNDNAME);
        if (::SymInitialize(::GetCurrentProcess(), nullptr, true)) {
          ready_ = true;
        }
#endif
#ifdef __GNUC__
        if (!bfd_) {
          return;
        }

        bfd_check_format(bfd_.get(), bfd_object);

        int const max_num_symbols = bfd_get_symtab_upper_bound(bfd_.get());
        if (max_num_symbols <= 0) {
          return;
        }

        symbols_.resize(max_num_symbols);
        int const num_symbols = bfd_canonicalize_symtab(bfd_.get(), symbols_.data());
        if (num_symbols <= 0) {
          return;
        }

        section_ = bfd_get_section_by_name(bfd_.get(), ".debug_info");
        if (!section_) {
          return;
        }

        ready_ = true;
#endif
      }

      inline ~address_to_symbol()
      {
#ifdef _WIN32
        if (ready_) {
          ::SymCleanup(::GetCurrentProcess());
          ready_ = false;
        }
#endif
      }

#if defined(_WIN32) && !defined(NDEBUG)
      std::ostream & operator () (std::ostream & out, void const * address) const
      {
        HANDLE const hProcess = ::GetCurrentProcess();
        DWORD64 const addr = reinterpret_cast<DWORD64>(address);

        // ファイル名と行数の取得
        {
          DWORD displacement = 0;

          IMAGEHLP_LINE64 file = { sizeof(IMAGEHLP_LINE64) };
          if (ready_ && ::SymGetLineFromAddr64(hProcess, addr, &displacement, &file)) {
            out << file.FileName << '(' << file.LineNumber << "): ";
          }
          else {
            out << "unknown(-1): ";
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
          if (ready_ && ::SymGetSymFromAddr64(hProcess, addr, &displacement, psymbol)) {
            out << psymbol->Name;
          } else {
            out << "unknown";
          }
        }

        return out;
      }
#else
      std::ostream & operator () (std::ostream & out, void const * address)
#ifdef _WIN32
        const
#endif
      {
#ifdef _WIN32
        out << std::ios::hex << std::setw(16) << address << " ";

        HANDLE const hProcess = ::GetCurrentProcess();
        DWORD64 const addr = reinterpret_cast<DWORD64>(address);
#endif
#ifdef __GNUC__
        Dl_info info;
        dladdr(address, &info);
        out << std::setw(16) << info.dli_saddr << " ";
#endif

        // モジュール名の取得
#ifdef _WIN32
        {
          IMAGEHLP_MODULE64 module = {sizeof(IMAGEHLP_MODULE64)};
          if (ready_ && ::SymGetModuleInfo64(hProcess, addr, &module)) {
            out << '<' << std::setw(8) << module.ModuleName << "> ";
          } else {
            out << "<unknown  > ";
          }
        }
#endif
#ifdef __GNUC__
        {
          out << '<' << std::setw(20) << (info.dli_sname ? info.dli_sname : "unknown") << "> ";
        }
#endif

        // 関数名の取得
#ifdef _WIN32
        {
          // シンボル名を格納するためのバッファを確保
          char symbol_buffer[sizeof(IMAGEHLP_SYMBOL64)+MAX_PATH];
          std::memset(symbol_buffer, 0, sizeof(symbol_buffer));

          IMAGEHLP_SYMBOL64 * psymbol = reinterpret_cast<IMAGEHLP_SYMBOL64 *>(symbol_buffer);
          psymbol->SizeOfStruct = sizeof(IMAGEHLP_SYMBOL64);
          psymbol->MaxNameLength = MAX_PATH;

          DWORD64 displacement = 0;
          if (ready_ && ::SymGetSymFromAddr64(hProcess, addr, &displacement, psymbol)) {
            out << psymbol->Name;
          } else {
            out << "unknown";
          }
        }
#endif
#ifdef __GNUC__
        int status = 0;
        std::shared_ptr<char> demangled(abi::__cxa_demangle(info.dli_sname, 0, 0, &status), std::free);

        char const * filename = NULL;
        char const * funcname = NULL;
        unsigned int line_num = 0;
        int const found = ready_ ?
          bfd_find_nearest_line(bfd_.get(), section_, symbols_.data(),
            reinterpret_cast<long>(address), &filename, &funcname, &line_num) : 0;

        out << ((found && funcname) ? funcname : "unknown");
#endif

        // ファイル名と行数の取得
#ifdef _WIN32
        {
          DWORD displacement = 0;

          IMAGEHLP_LINE64 file = {sizeof(IMAGEHLP_LINE64)};
          if (ready_ && ::SymGetLineFromAddr64(hProcess, addr, &displacement, &file)) {
            out << '(' << reactor::utility::path_to_filename(file.FileName) << ':' << file.LineNumber << ')';
          } else {
            out << "(unknown:-1)";
          }
        }
#endif
#ifdef __GNUC__
        if (found && filename) {
          out << '(' << reactor::utility::path_to_filename(filename) << ':' << line_num << ')';
        } else {
          out << "(unknown:-1)";
        }
#endif
        return out;
      }
#endif

    public:
      inline static std::ostream & convert(std::ostream & out, void const * address)
      {
        static address_to_symbol _;
        return _(out, address);
      }

    private:
      bool ready_;

#ifdef __GNUC__
      std::shared_ptr<bfd>  bfd_;
      std::vector<asymbol*> symbols_;
      asection *            section_;
#endif
    };
#endif

#ifdef _WIN32
    inline int backtrace(void ** stacktrace, int const depth)
    {
      return ::RtlCaptureStackBackTrace(0, depth, stacktrace, nullptr);
    }
#endif

    std::string stack_trace()
    {
#if defined(_WIN32) || defined(__GNUC__)
#define MAX_STACKTRACE_DEPTH 63
      void * stacktrace[MAX_STACKTRACE_DEPTH];  // the number of frames must be less than 63.
      int const size = backtrace(stacktrace, MAX_STACKTRACE_DEPTH);

      std::ostringstream out;
      for (int i = 0; i < size; ++i) {
        reactor::utility::address_to_symbol::convert(out, stacktrace[i]) << '\n';
      }
      return out.str();
#else
      return "\n";
#endif
    }
  }
}

#endif//REACTOR_UTILITY_STACKTRACE_HPP_INCLUDED
