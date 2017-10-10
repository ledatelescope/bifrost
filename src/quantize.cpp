/*
 * Copyright (c) 2016, The Bifrost Authors. All rights reserved.
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

#include <iostream>

#ifdef BF_CUDA_ENABLED
#include <guantize.hu>
#endif

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

inline int8_t clip_4bit(int8_t x) {
	return min(max(x,int8_t(-7)),int8_t(7));
}

template<typename F>
inline F clip_2bit(F x) {
	return min(max(x,F(-1)),F(1));
}

template<typename F>
inline F clip_1bit(F x) {
	return x >= F(0) ? F(1) : F(0);
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

template<typename T, typename Func, typename Size>
void foreach_simple_cpu_4bit(T const* in,
                             int8_t*       out,
                             Size     nelement,
                             Func     func) {
	T tempR;
	T tempI;
	int8_t tempO;
	for( Size i=0; i<nelement; i+=2 ) {
		tempR = in[i+0];
		tempI = in[i+1];
		if(func.byteswap_in) {
			byteswap(tempR, &tempR);
			byteswap(tempI, &tempI);
		}
		//std::cout << tempR << ", " << tempI << " --> " << rint(clip_4bit(tempR)) << ", " << rint(clip_4bit(tempI)) << '\n';
		tempO = (((int8_t(rint(clip_4bit(tempR*func.scale)))*16)     ) & 0xF0) | \
			    (((int8_t(rint(clip_4bit(tempI*func.scale)))*16) >> 4) & 0x0F);
		if(func.byteswap_out) {
			byteswap(tempO, &tempO);
		}
		out[i/2] = tempO;
	}
}

template<typename T, typename Func, typename Size>
void foreach_simple_cpu_2bit(T const* in,
                             int8_t*  out,
                             Size     nelement,
                             Func     func) {
	T tempA;
	T tempB;
	T tempC;
	T tempD;
	int8_t tempO;
	for( Size i=0; i<nelement; i+=4 ) {
		tempA = in[i+0];
		tempB = in[i+1];
		tempC = in[i+2];
		tempD = in[i+3];
		if(func.byteswap_in) {
			byteswap(tempA, &tempA);
			byteswap(tempB, &tempB);
			byteswap(tempC, &tempC);
			byteswap(tempD, &tempD);
		}
		//std::cout << tempR << ", " << tempI << " --> " << rint(clip_4bit(tempR)) << ", " << rint(clip_4bit(tempI)) << '\n';
		tempO = (((int8_t(rint(clip_2bit(tempA*func.scale)))*64)     ) & 0xC0) | \
		(((int8_t(rint(clip_2bit(tempB*func.scale)))*64) >> 2) & 0x30) | \
		(((int8_t(rint(clip_2bit(tempC*func.scale)))*64) >> 4) & 0x0C) | \
		(((int8_t(rint(clip_2bit(tempD*func.scale)))*64) >> 6) & 0x03);
		if(func.byteswap_out) {
			byteswap(tempO, &tempO);
		}
		out[i/4] = tempO;
	}
}

template<typename T, typename Func, typename Size>
void foreach_simple_cpu_1bit(T const* in,
                             int8_t*  out,
                             Size     nelement,
                             Func     func) {
	T tempA;
	T tempB;
	T tempC;
	T tempD;
	T tempE;
	T tempF;
	T tempG;
	T tempH;
	int8_t tempO;
	for( Size i=0; i<nelement; i+=8 ) {
		tempA = in[i+0];
		tempB = in[i+1];
		tempC = in[i+2];
		tempD = in[i+3];
		tempE = in[i+4];
		tempF = in[i+5];
		tempG = in[i+6];
		tempH = in[i+7];
		if(func.byteswap_in) {
			byteswap(tempA, &tempA);
			byteswap(tempB, &tempB);
			byteswap(tempC, &tempC);
			byteswap(tempD, &tempD);
			byteswap(tempE, &tempE);
			byteswap(tempF, &tempF);
			byteswap(tempG, &tempG);
			byteswap(tempH, &tempH);
		}
		//std::cout << tempR << ", " << tempI << " --> " << rint(clip_4bit(tempR)) << ", " << rint(clip_4bit(tempI)) << '\n';
		tempO = (((int8_t(rint(clip_1bit(tempA*func.scale)))*128)     ) & 0x08) | \
		(((int8_t(rint(clip_1bit(tempB*func.scale)))*128) >> 1) & 0x04) | \
		(((int8_t(rint(clip_1bit(tempC*func.scale)))*128) >> 2) & 0x02) | \
		(((int8_t(rint(clip_1bit(tempD*func.scale)))*128) >> 3) & 0x10) | \
		(((int8_t(rint(clip_1bit(tempE*func.scale)))*128) >> 4) & 0x08) | \
		(((int8_t(rint(clip_1bit(tempF*func.scale)))*128) >> 5) & 0x04) | \
		(((int8_t(rint(clip_1bit(tempG*func.scale)))*128) >> 6) & 0x02) | \
		(((int8_t(rint(clip_1bit(tempH*func.scale)))*128) >> 7) & 0x01);
		if(func.byteswap_out) {
			byteswap(tempO, &tempO);
		}
		out[i/8] = tempO;
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
	
#ifdef BF_CUDA_ENABLED
	BF_ASSERT(space_accessible_from(in->space, BF_SPACE_SYSTEM) || (space_accessible_from(in->space, BF_SPACE_CUDA) && space_accessible_from(out->space, BF_SPACE_CUDA)),
	          BF_STATUS_UNSUPPORTED_SPACE);
	BF_ASSERT(space_accessible_from(out->space, BF_SPACE_SYSTEM) || (space_accessible_from(in->space, BF_SPACE_CUDA) && space_accessible_from(out->space, BF_SPACE_CUDA)),
	          BF_STATUS_UNSUPPORTED_SPACE);
#else
	BF_ASSERT(space_accessible_from(in->space, BF_SPACE_SYSTEM),
	          BF_STATUS_UNSUPPORTED_SPACE);
	BF_ASSERT(space_accessible_from(out->space, BF_SPACE_SYSTEM),
	          BF_STATUS_UNSUPPORTED_SPACE);
#endif
	
	size_t nelement = num_contiguous_elements(in);
	bool byteswap_in  = ( in->big_endian != is_big_endian());
	bool byteswap_out = (out->big_endian != is_big_endian());
	
#ifdef BF_CUDA_ENABLED
	if( space_accessible_from(in->space, BF_SPACE_CUDA) ) {
		BF_ASSERT(nelement<=(size_t)512*65535*65535, BF_STATUS_UNSUPPORTED_SHAPE);
	}
#endif
	
#define CALL_FOREACH_SIMPLE_CPU_QUANTIZE(itype,stype,otype) \
	foreach_simple_cpu((itype*)in->data, \
	                   (otype*)out->data, \
	                   nelement, \
	                   QuantizeFunctor<itype,stype,otype> \
	                   (scale,byteswap_in,byteswap_out))
	
#ifdef BF_CUDA_ENABLED
#define CALL_FOREACH_SIMPLE_GPU_QUANTIZE(itype,stype,otype) \
	launch_foreach_simple_gpu((itype*)in->data, \
	                          (otype*)out->data, \
	                          nelement, \
	                          GuantizeFunctor<itype,stype,otype> \
	                          (scale,byteswap_in,byteswap_out), \
	                          (cudaStream_t)0)
#endif
	
	// **TODO: Need CF32 --> CI* separately to support conjugation
	if( in->dtype == BF_DTYPE_F32 || in->dtype == BF_DTYPE_CF32 ) {
		// TODO: Support T-->T with endian conversion (like quantize but with identity func instead)
		switch( out->dtype ) {
		case BF_DTYPE_CI1: nelement *= 2;
		case BF_DTYPE_I1: {
			BF_ASSERT(nelement % 8 == 0, BF_STATUS_INVALID_SHAPE);
#ifdef BF_CUDA_ENABLED
			if( space_accessible_from(in->space, BF_SPACE_CUDA) ) {
				launch_foreach_simple_gpu_1bit((float*)in->data, \
				                               (int8_t*)out->data, \
				                               nelement, \
				                               GuantizeFunctor<float,float,uint8_t> \
				                               (scale,byteswap_in,byteswap_out), \
				                               (cudaStream_t)0);
			} else {
				foreach_simple_cpu_1bit((float*)in->data, \
				                        (int8_t*)out->data, \
				                        nelement, \
				                        QuantizeFunctor<float,float,uint8_t> \
				                        (scale,byteswap_in,byteswap_out));
			}
#else
			foreach_simple_cpu_1bit((float*)in->data, \
				                   (int8_t*)out->data, \
				                   nelement, \
				                   QuantizeFunctor<float,float,uint8_t> \
				                   (scale,byteswap_in,byteswap_out));
#endif
			break;
		}
		case BF_DTYPE_CI2: nelement *= 2;
		case BF_DTYPE_I2: {
			BF_ASSERT(nelement % 4 == 0, BF_STATUS_INVALID_SHAPE);
#ifdef BF_CUDA_ENABLED
			if( space_accessible_from(in->space, BF_SPACE_CUDA) ) {
				launch_foreach_simple_gpu_2bit((float*)in->data, \
				                               (int8_t*)out->data, \
				                               nelement, \
				                               GuantizeFunctor<float,float,uint8_t> \
				                               (scale,byteswap_in,byteswap_out), \
				                               (cudaStream_t)0);
			} else {
				foreach_simple_cpu_2bit((float*)in->data, \
				                        (int8_t*)out->data, \
				                        nelement, \
				                        QuantizeFunctor<float,float,uint8_t> \
				                        (scale,byteswap_in,byteswap_out));
			}
#else
			foreach_simple_cpu_2bit((float*)in->data, \
				                   (int8_t*)out->data, \
				                   nelement, \
				                   QuantizeFunctor<float,float,uint8_t> \
				                   (scale,byteswap_in,byteswap_out));
#endif
			break;
		}
		case BF_DTYPE_CI4: nelement *= 2;
		case BF_DTYPE_I4: {
			BF_ASSERT(nelement % 2 == 0, BF_STATUS_INVALID_SHAPE);
#ifdef BF_CUDA_ENABLED
			if( space_accessible_from(in->space, BF_SPACE_CUDA) ) {
				launch_foreach_simple_gpu_4bit((float*)in->data, \
				                               (int8_t*)out->data, \
				                               nelement, \
				                               GuantizeFunctor<float,float,uint8_t> \
				                               (scale,byteswap_in,byteswap_out), \
				                               (cudaStream_t)0);
			} else {
				foreach_simple_cpu_4bit((float*)in->data, \
				                        (int8_t*)out->data, \
				                        nelement, \
				                        QuantizeFunctor<float,float,uint8_t> \
				                        (scale,byteswap_in,byteswap_out));
			}
#else
			foreach_simple_cpu_4bit((float*)in->data, \
				                   (int8_t*)out->data, \
				                   nelement, \
				                   QuantizeFunctor<float,float,uint8_t> \
				                   (scale,byteswap_in,byteswap_out));
#endif
			break;
		}
		case BF_DTYPE_CI8: nelement *= 2;
		case BF_DTYPE_I8: {
#ifdef BF_CUDA_ENABLED
			if( space_accessible_from(in->space, BF_SPACE_CUDA) ) {
				CALL_FOREACH_SIMPLE_GPU_QUANTIZE(float,float,int8_t);
			} else {
				CALL_FOREACH_SIMPLE_CPU_QUANTIZE(float,float,int8_t);
			}
#else
			CALL_FOREACH_SIMPLE_CPU_QUANTIZE(float,float,int8_t);
#endif
			break;
		}
		case BF_DTYPE_CI16: nelement *= 2;
		case BF_DTYPE_I16: {
#ifdef BF_CUDA_ENABLED
			if( space_accessible_from(in->space, BF_SPACE_CUDA) ) {
				CALL_FOREACH_SIMPLE_GPU_QUANTIZE(float,float,int16_t);
			} else {
				CALL_FOREACH_SIMPLE_CPU_QUANTIZE(float,float,int16_t);
			}
#else
			CALL_FOREACH_SIMPLE_CPU_QUANTIZE(float,float,int16_t);
#endif
			break;
		}
		case BF_DTYPE_CI32: nelement *= 2;
		case BF_DTYPE_I32: {
#ifdef BF_CUDA_ENABLED
			if( space_accessible_from(in->space, BF_SPACE_CUDA) ) {
				CALL_FOREACH_SIMPLE_GPU_QUANTIZE(float,double,int32_t);
			} else {
				CALL_FOREACH_SIMPLE_CPU_QUANTIZE(float,double,int32_t);
			}
#else
			CALL_FOREACH_SIMPLE_CPU_QUANTIZE(float,double,int32_t);
#endif
			break;
		}
		case BF_DTYPE_U8: {
#ifdef BF_CUDA_ENABLED
			if( space_accessible_from(in->space, BF_SPACE_CUDA) ) {
				
			} else {
				CALL_FOREACH_SIMPLE_CPU_QUANTIZE(float,float,uint8_t);
			}
#else
			CALL_FOREACH_SIMPLE_CPU_QUANTIZE(float,float,uint8_t);
#endif
			break;
		}
		case BF_DTYPE_U16: {
#ifdef BF_CUDA_ENABLED
			if( space_accessible_from(in->space, BF_SPACE_CUDA) ) {
				CALL_FOREACH_SIMPLE_GPU_QUANTIZE(float,float,uint16_t);
			} else {
				CALL_FOREACH_SIMPLE_CPU_QUANTIZE(float,float,uint16_t);
			}
#else
			CALL_FOREACH_SIMPLE_CPU_QUANTIZE(float,float,uint16_t);
#endif
			break;
		}
		case BF_DTYPE_U32: {
#ifdef BF_CUDA_ENABLED
			if( space_accessible_from(in->space, BF_SPACE_CUDA) ) {
				CALL_FOREACH_SIMPLE_GPU_QUANTIZE(float,double,uint32_t);
			} else {
				CALL_FOREACH_SIMPLE_CPU_QUANTIZE(float,double,uint32_t);
			}
#else
			CALL_FOREACH_SIMPLE_CPU_QUANTIZE(float,double,uint32_t);
#endif
			break;
		}
		default: BF_FAIL("Supported bfQuantize output dtype", BF_STATUS_UNSUPPORTED_DTYPE);
		}
	} else {
		BF_FAIL("Supported bfQuantize input dtype", BF_STATUS_UNSUPPORTED_DTYPE);
	}
#undef CALL_FOREACH_SIMPLE_CPU_QUANTIZE
#undef CALL_FOREACH_SIMPLE_GPU_QUANTIZE
	return BF_STATUS_SUCCESS;
}
