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

#include <bifrost/quantize.h>
#include "utils.hpp"

#include <limits>
#include <cmath>

using std::max;
using std::min;

// Note: maxval is always the max representable integer value
//         E.g., 8-bit => 127 (signed), 255 (unsigned)
//       minval is either -maxval (signed) or 0 (unsigned)
//       The min representable value of signed integers is not used
//         E.g., int8_t => -128 is not used
template<typename T>
inline T maxval(T x=T()) { return std::numeric_limits<T>::max(); }
template<typename T>
inline typename std::enable_if<!std::is_signed<T>::value,T>::type
minval(T x=T()) { return T(0); }
template<typename T>
inline typename std::enable_if< std::is_signed<T>::value,T>::type
minval(T x=T()) { return -maxval<T>(); }

template<typename I, typename F>
inline F clip(F x) {
	return min(max(x,F(minval<I>())),F(maxval<I>()));
}

template<typename IType, typename SType, typename OType>
inline void quantize(IType ival, SType scale, OType& oval) {
	//std::cout << (int)minval<OType>() << ", " << (int)maxval<OType>() << std::endl;
	//std::cout << scale << std::endl;
	//std::cout << ival
	//          << " --> " << ival*scale
	//          << " --> " << clip<OType>(ival*scale)
	//          << " --> " << rint(clip<OType>(ival*scale))
	//          << " --> " << (int)OType(rint(clip<OType>(ival*scale)))
	//          << std::endl;
	oval = OType(rint(clip<OType>(ival*scale)));
}

template<typename IType, typename SType, typename OType>
struct QuantizeFunctor {
	SType scale;
	bool  byteswap_in;
	bool  byteswap_out;
	QuantizeFunctor(SType scale_, bool byteswap_in_, bool byteswap_out_)
		: scale(scale_),
		  byteswap_in(byteswap_in_),
		  byteswap_out(byteswap_out_) {}
	void operator()(IType ival, OType& oval) const {
		if( byteswap_in ) {
			byteswap(ival, &ival);
		}
		quantize(ival, scale, oval);
		if( byteswap_out ) {
			byteswap(oval, &oval);
		}
	}
};

template<typename T, typename U, typename Func, typename Size>
void foreach_simple_cpu(T const* in,
                        U*       out,
                        Size     nelement,
                        Func     func) {
	for( Size i=0; i<nelement; ++i ) {
		func(in[i], out[i]);
		//std::cout << std::hex << (int)in[i] << " --> " << (int)out[i] << std::endl;
	}
}

BFstatus bfQuantize(BFarray const* in,
                    BFarray const* out,
                    double         scale) {
	BF_ASSERT(in,  BF_STATUS_INVALID_POINTER);
	BF_ASSERT(out, BF_STATUS_INVALID_POINTER);
	BF_ASSERT(!out->immutable, BF_STATUS_INVALID_POINTER);
	BF_ASSERT(shapes_equal(in, out), BF_STATUS_INVALID_SHAPE);
	BF_ASSERT(BF_DTYPE_IS_COMPLEX( in->dtype) ==
	          BF_DTYPE_IS_COMPLEX(out->dtype),
	          BF_STATUS_INVALID_DTYPE);
	BF_ASSERT(BF_DTYPE_IS_COMPLEX(in->dtype) || !in->conjugated,
	          BF_STATUS_INVALID_DTYPE);
	BF_ASSERT(BF_DTYPE_IS_COMPLEX(out->dtype) || !in->conjugated,
	          BF_STATUS_INVALID_DTYPE);
	
	// TODO: Support conjugation
	BF_ASSERT((!BF_DTYPE_IS_COMPLEX(in->dtype)) ||
	          (in->conjugated == out->conjugated),
	          BF_STATUS_UNSUPPORTED);
	
	// TODO: Support padded arrays
	BF_ASSERT(is_contiguous(in),  BF_STATUS_UNSUPPORTED_STRIDE);
	BF_ASSERT(is_contiguous(out), BF_STATUS_UNSUPPORTED_STRIDE);
	
	// TODO: Support CUDA space
	BF_ASSERT(space_accessible_from(in->space, BF_SPACE_SYSTEM),
	          BF_STATUS_UNSUPPORTED_SPACE);
	BF_ASSERT(space_accessible_from(out->space, BF_SPACE_SYSTEM),
	          BF_STATUS_UNSUPPORTED_SPACE);
	
	size_t nelement = num_contiguous_elements(in);
	bool byteswap_in  = ( in->big_endian != is_big_endian());
	bool byteswap_out = (out->big_endian != is_big_endian());
	
#define CALL_FOREACH_SIMPLE_CPU_QUANTIZE(itype,stype,otype) \
	foreach_simple_cpu((itype*)in->data, \
	                   (otype*)out->data, \
	                   nelement, \
	                   QuantizeFunctor<itype,stype,otype> \
	                   (scale,byteswap_in,byteswap_out))
	
	// **TODO: Need CF32 --> CI* separately to support conjugation
	if( in->dtype == BF_DTYPE_F32 || in->dtype == BF_DTYPE_CF32 ) {
		// TODO: Support T-->T with endian conversion (like quantize but with identity func instead)
		switch( out->dtype ) {
		case BF_DTYPE_CI8: nelement *= 2;
		case BF_DTYPE_I8: {
			CALL_FOREACH_SIMPLE_CPU_QUANTIZE(float,float,int8_t); break;
		}
		case BF_DTYPE_CI16: nelement *= 2;
		case BF_DTYPE_I16: {
			CALL_FOREACH_SIMPLE_CPU_QUANTIZE(float,float,int16_t); break;
		}
		case BF_DTYPE_CI32: nelement *= 2;
		case BF_DTYPE_I32: {
			CALL_FOREACH_SIMPLE_CPU_QUANTIZE(float,double,int32_t); break;
		}
		case BF_DTYPE_U8: {
			CALL_FOREACH_SIMPLE_CPU_QUANTIZE(float,float,uint8_t); break;
		}
		case BF_DTYPE_U16: {
			CALL_FOREACH_SIMPLE_CPU_QUANTIZE(float,float,uint16_t); break;
		}
		case BF_DTYPE_U32: {
			CALL_FOREACH_SIMPLE_CPU_QUANTIZE(float,double,uint32_t); break;
		}
		default: BF_FAIL("Supported bfQuantize output dtype", BF_STATUS_UNSUPPORTED_DTYPE);
		}
	} else {
		BF_FAIL("Supported bfQuantize input dtype", BF_STATUS_UNSUPPORTED_DTYPE);
	}
#undef CALL_FOREACH_SIMPLE_CPU_QUANTIZE
	return BF_STATUS_SUCCESS;
}
