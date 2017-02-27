/*
 * Copyright (c) 2016, The Bifrost Authors. All rights reserved.
 * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * * Redistributions of source code must retain the above copyright
 *   notice, this list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above copyright
 *   notice, this list of conditions and the following disclaimer in the
 *   documentation and/or other materials provided with the distribution.
 * * Neither the name of The Bifrost Authors nor the names of its
 *   contributors may be used to endorse or promote products derived
 *   from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#ifndef __CUDACC_RTC__
#include <cmath>
using std::atan2;
#endif // __CUDACC_RTC__

// Storage-only formats
template<typename Real, class Enable=void>
struct Complex;

template<typename Real, typename PromotedReal=float>
struct __attribute__((aligned(2*sizeof(Real)))) ComplexBase {
	typedef Real real_type;
	union { real_type x, real; };
	union { real_type y, imag; };
	operator Complex<PromotedReal>();
};
template<>
struct __attribute__((aligned(2))) Complex<signed char>
	: public ComplexBase<signed char> {};
template<>
struct __attribute__((aligned(4))) Complex<short>
	: public ComplexBase<short> {};
template<>
struct __attribute__((aligned(8))) Complex<int>
	: public ComplexBase<int,double> {};

typedef Complex<float>  Complex32;
typedef Complex<double> Complex64;

template<typename R> inline __host__ __device__ Complex<R> exp(Complex<R> const& a) {
	return exp(a.x) * Complex<R>(cos(a.y), sin(a.y));
}

// TODO: Move these somewhere else
template<typename F>
inline __host__ __device__
void quantize(F f, signed char* q) {
	*q = max(min(rint(f), +127), -127);
}
template<typename F>
inline __host__ __device__
void quantize(F f, unsigned char* q) {
	*q = max(min(rint(f), +255), 0);
}
template<typename F>
inline __host__ __device__
void quantize(F f, signed short* q) {
	*q = max(min(rint(f), +32767), -32767);
}
template<typename F>
inline __host__ __device__
void quantize(F f, unsigned short* q) {
	*q = max(min(rint(f), +65535), 0);
}

template<typename F, typename Q>
inline __host__ __device__
void quantize(Complex<F> const& c, Complex<Q>* q) {
	quantize(c.x, &q->x);
	quantize(c.y, &q->y);
}

namespace Complex_detail {
// TODO: De-dupe this with the one in ArrayIndexer.cuh
template<bool B, class T = void>
struct enable_if {};
template<class T>
struct enable_if<true, T> { typedef T type; };

template<typename T> struct is_floating_point    { enum { value = false }; };
template<> struct is_floating_point<float>       { enum { value = true  }; };
template<> struct is_floating_point<double>      { enum { value = true  }; };
template<> struct is_floating_point<long double> { enum { value = true  }; };

#ifdef __CUDACC__
template<typename Real> struct cuda_vector2_type {};
template<>              struct cuda_vector2_type<float>  { typedef float2  type; };
template<>              struct cuda_vector2_type<double> { typedef double2 type; };
#endif

} // namespace Complex_detail

template<typename T>
struct __attribute__((aligned(sizeof(T)*2)))
Complex<T, typename Complex_detail::enable_if<Complex_detail::is_floating_point<T>::value>::type> {
	// Note: Using unions here may prevent vectorized load/store
	typedef T real_type;
	//real_type x, y;
	union { real_type x, real; };
	union { real_type y, imag; };
	
	inline __host__ __device__ Complex() {}//: x(0), y(0) {}
	inline __host__ __device__ Complex(real_type x_, real_type y_=0) : x(x_), y(y_) {}
	// Implicit conversion from low-precision storage types
	inline __host__ __device__ Complex(Complex<signed char> c) : x(c.x), y(c.y) {}
	inline __host__ __device__ Complex(Complex<short> c)       : x(c.x), y(c.y) {}
#ifdef __CUDACC__
	// Note: Use float2 to ensure vectorized load/store
	inline __host__ __device__ Complex(typename Complex_detail::cuda_vector2_type<T>::type c) : x(c.x), y(c.y) {}
	inline __host__ __device__ operator typename Complex_detail::cuda_vector2_type<T>::type() const { return make_float2(x,y); }
#endif
	inline __host__ __device__ Complex& assign(real_type x_, real_type y_) { x = x_; y = y_; return *this; }
	inline __host__ __device__ Complex& operator+=(Complex c) { x += c.x; y += c.y; return *this; }
	inline __host__ __device__ Complex& operator-=(Complex c) { x -= c.x; y -= c.y; return *this; }
	inline __host__ __device__ Complex& operator*=(Complex c) {
		Complex tmp;
		tmp.x  = x*c.x;
		tmp.x -= y*c.y;
		tmp.y  = y*c.x;
		tmp.y += x*c.y;
		return *this = tmp;
	}
	inline __host__ __device__ Complex& operator/=(Complex c) {
		return *this *= c.conj() / c.mag2();
	}
	inline __host__ __device__ Complex& operator*=(real_type s) { x *= s; y *= s; return *this; }
	inline __host__ __device__ Complex& operator/=(real_type s) { return *this *= 1/s; }
	inline __host__ __device__ Complex  operator+() const { return Complex(+x,+y); }
	inline __host__ __device__ Complex  operator-() const { return Complex(-x,-y); }
	inline __host__ __device__ Complex   conj()  const { return Complex(x, -y); }
	//inline __host__ __device__ real_type real()  const { return x; }
	//inline __host__ __device__ real_type imag()  const { return y; }
	//inline __host__ __device__ real_type& real()       { return x; }
	//inline __host__ __device__ real_type& imag()       { return y; }
	//inline __host__ __device__ void      real(real_type x_) { x = x_; }
	//inline __host__ __device__ void      imag(real_type y_) { y = y_; }
	inline __host__ __device__ real_type phase() const { return atan2(y, x); }
	inline __host__ __device__ real_type mag2()  const { real_type a = x*x; a += y*y; return a; }
	inline __host__ __device__ real_type mag()   const { return sqrt(this->mag2()); }
	inline __host__ __device__ real_type abs()   const { return this->mag(); }
	inline __host__ __device__ Complex&  mad(Complex a, Complex b) {
		x += a.x*b.x;
		x -= a.y*b.y;
		y += a.y*b.x;
		y += a.x*b.y;
		return *this;
	}
	inline __host__ __device__ Complex& msub(Complex a, Complex b) {
		x -= a.x*b.x;
		x += a.y*b.y;
		y -= a.y*b.x;
		y -= a.x*b.y;
		return *this;
	}
	inline __host__ __device__ bool operator==(Complex const& c) const { return (x==c.x) && (y==c.y); }
	inline __host__ __device__ bool operator!=(Complex const& c) const { return !(*this == c); }
	inline __host__ __device__ bool isreal(real_type tol=1e-6) const {
		return y/x <= tol;
	}
};

#define DEFINE_BINARY_OPERATOR(op) \
template<typename T> \
inline __host__ __device__ \
Complex<T> operator op(Complex<T> a, Complex<T> b) { \
	Complex<T> c = a; \
	c op##= b; \
	return c; \
} \
template<typename T, typename U> \
inline __host__ __device__ \
Complex<T> operator op(Complex<T> a, U b) { \
	Complex<T> c = a; \
	c op##= b; \
	return c; \
} \
template<typename T, typename U> \
inline __host__ __device__ \
Complex<T> operator op(U a, Complex<T> b) { \
	Complex<T> c = a; \
	c op##= b; \
	return c; \
}
DEFINE_BINARY_OPERATOR(+)
DEFINE_BINARY_OPERATOR(-)
DEFINE_BINARY_OPERATOR(*)
DEFINE_BINARY_OPERATOR(/)
#undef DEFINE_BINARY_OPERATOR

template<typename Real, typename PromotedReal>
ComplexBase<Real,PromotedReal>::operator Complex<PromotedReal>() {
	Complex<PromotedReal> c = {real, imag};
	return c;
}

