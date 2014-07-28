#include <iostream>
#include <Windows.h>

#include <reactor/cuda/context.hpp>
#include <reactor/cuda/memory.hpp>
#include <reactor/cuda/stream.hpp>

__global__ void da(int * const a) { *a = 3; }
__global__ void db(int * const b) { *b = 7; }

__global__ void dc(int const * const a, int const * const b, int * const c)
{
	*c = *a + *b;
}

__global__ void dd(int const * const a, int const * const b, int * const d)
{
	*d = *a * *b;
}

int main()
try
{
	std::ios::sync_with_stdio(false);
	{
		reactor::cuda::context _;
		{
#if 0
			reactor::cuda::stream s1;
			reactor::cuda::stream s2;
			reactor::cuda::stream s3;
			reactor::cuda::stream s4;

			reactor::cuda::event e1;
			reactor::cuda::event e2;

			auto d_a = reactor::cuda::malloc<int>(1);
			auto d_b = reactor::cuda::malloc<int>(1);
			auto d_c = reactor::cuda::malloc<int>(1);
			auto d_d = reactor::cuda::malloc<int>(1);

			int a = -1;
			int b = -1;
			int c = -1;
			int d = -1;

			{
				da<<<1, 1, 0, s1.get()>>>(d_a.get());
				s1.notify(e1);
				reactor::cuda::memcpy(&a, d_a.get(), 1, s1.get());
			}
			{
				db<<<1, 1, 0, s2.get()>>>(d_b.get());
				s2.notify(e2);
				reactor::cuda::memcpy(&b, d_b.get(), 1, s2.get());
			}
			{
				s3.wait(e1);
				s3.wait(e2);
				dc<<<1, 1, 0, s3.get()>>>(d_a.get(), d_b.get(), d_c.get());
				reactor::cuda::memcpy(&c, d_c.get(), 1, s3.get());
			}
			{
				s4.wait(e1);
				s4.wait(e2);
				dd<<<1, 1, 0, s4.get()>>>(d_a.get(), d_b.get(), d_d.get());
				reactor::cuda::memcpy(&d, d_d.get(), 1, s4.get());
			}
			{
				s1.synchronize();
				s2.synchronize();
				s3.synchronize();
				std::cout << a << " + " << b << " = " << c << '\n';
			}
			{
				s1.synchronize();
				s2.synchronize();
				s4.synchronize();
				std::cout << a << " * " << b << " = " << d << '\n';
			}
#else
			reactor::cuda::stream sc; // compuate-stream
			reactor::cuda::stream st; // transfer-stream

			reactor::cuda::event ea;
			reactor::cuda::event eb;
			reactor::cuda::event ec;
			reactor::cuda::event ed;

			auto d_a = reactor::cuda::malloc<int>(1);
			auto d_b = reactor::cuda::malloc<int>(1);
			auto d_c = reactor::cuda::malloc<int>(1);
			auto d_d = reactor::cuda::malloc<int>(1);

			int a = -1;
			int b = -1;
			int c = -1;
			int d = -1;

			da<<<1, 1, 0, sc.get()>>>(d_a.get()                      ); sc.notify(ea);
			db<<<1, 1, 0, sc.get()>>>(d_b.get()                      ); sc.notify(eb);
			dc<<<1, 1, 0, sc.get()>>>(d_a.get(), d_b.get(), d_c.get()); sc.notify(ec);
			dd<<<1, 1, 0, sc.get()>>>(d_a.get(), d_b.get(), d_d.get()); sc.notify(ed);

			st.wait(ea); reactor::cuda::memcpy(&a, d_a.get(), 1, st.get());
			st.wait(eb); reactor::cuda::memcpy(&b, d_b.get(), 1, st.get());
			st.wait(ec); reactor::cuda::memcpy(&c, d_c.get(), 1, st.get());
			st.wait(ed); reactor::cuda::memcpy(&d, d_d.get(), 1, st.get());

			st.synchronize();
			std::cout << a << " + " << b << " = " << c << '\n';
			std::cout << a << " * " << b << " = " << d << '\n';
#endif
		}
	}
	return 0;
}
catch (reactor::cuda::exception const & e)
{
	std::cerr << e.what() << '\n';
}
