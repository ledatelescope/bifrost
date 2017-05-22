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

#include <bifrost/unpack.h>
#include "utils.hpp"

// sign_extend == true  => output has same value as input  (slower)
// sign_extend == false => output is scaled by 2**(8-nbit) (faster)

template<int NBIT, typename K, typename T>
inline void rshift_subwords(T& val) {
	for( int k=0; k<(int)(sizeof(T)/sizeof(K)); ++k ) {
		((K*)&val)[k] >>= NBIT;
	}
}
// 2x 4-bit --> 2x 8-bit (unsigned)
inline void unpack(uint8_t   ival,
                   uint16_t& oval,
                   bool      byte_reverse,
                   bool      align_msb,
                   bool      conjugate) {
	// Note: Ignores conjugate
	if( byte_reverse ) {
		// ........ABCDEFGH
		// EFGH....ABCD....
		oval = ival;
		oval = (oval | (oval << 12)) & 0xF0F0;
	} else {
		// ....ABCDEFGH....
		// ABCD....EFGH....
		oval = ival << 4;
		oval = (oval | (oval <<  4)) & 0xF0F0;
	}
	if( !align_msb ) {
		// >>>>ABCD>>>>EFGH
		oval >>= 4;
	}
}

// 4x 2-bit --> 4x 8-bit (unsigned)
inline void unpack(uint8_t   ival,
                   uint32_t& oval,
                   bool      byte_reverse,
                   bool      align_msb,
                   bool      conjugate) {
	// Note: Ignores conjugate
	// ..................ABCDEFGH......
	// ......ABCD............EFGH......
	// AB......CD......EF......GH......
	oval = ival << 6;
	oval = (oval | (oval << 12)) & 0x03C003C0;
	oval = (oval | (oval <<  6)) & 0xC0C0C0C0;
	if( byte_reverse) {
		byteswap(oval, &oval);
	}
	if( !align_msb ) {
		// >>>>>>AB>>>>>>CD>>>>>>EF>>>>>>GH
		oval >>= 6;
	}
}

// 8x 1-bit --> 8x 8-bit (unsigned)
inline void unpack(uint8_t   ival,
                   uint64_t& oval,
                   bool      byte_reverse,
                   bool      align_msb,
                   bool      conjugate) {
	// Note: Ignores conjugate
	// .................................................ABCDEFGH.......
	// .....................ABCD............................EFGH.......
	// .......AB..............CD..............EF..............GH.......
	// A.......B.......C.......D.......E.......F.......G.......H.......
	oval = ival << 7;
	oval = (oval | (oval << 28)) & 0x0000078000000780;
	oval = (oval | (oval << 14)) & 0x0180018001800180;
	oval = (oval | (oval <<  7)) & 0x8080808080808080;
	if( byte_reverse) {
		byteswap(oval, &oval);
	}
	if( !align_msb ) {
		// >>>>>>>A>>>>>>>B>>>>>>>C>>>>>>>D
		oval >>= 7;
	}
}

template<typename K, typename T>
inline void conjugate_subwords(T& val) {
	for( int k=1; k<(int)(sizeof(T)/sizeof(K)); k+=2 ) {
		K& val_imag = ((K*)&val)[k];
		val_imag = -val_imag;
	}
}

// 2x 4-bit --> 2x 8-bit (signed)
inline void unpack(uint8_t  ival,
                   int16_t& oval,
                   bool     byte_reverse,
                   bool     align_msb,
                   bool     conjugate) {
	if( byte_reverse ) {
		// ........ABCDEFGH
		// EFGH....ABCD....
		oval = ival;
		oval = (oval | (oval << 12)) & 0xF0F0;
	} else {
		// ....ABCDEFGH....
		// ABCD....EFGH....
		oval = ival << 4;
		oval = (oval | (oval <<  4)) & 0xF0F0;
	}
	if( !align_msb ) {
		// >>>>ABCD>>>>EFGH
		rshift_subwords<4,int8_t>(oval);
	}
	if( conjugate ) {
		conjugate_subwords<int8_t>(oval);
	}
}
// 4x 2-bit --> 4x 8-bit (signed)
inline void unpack(uint8_t  ival,
                   int32_t& oval,
                   bool     byte_reverse,
                   bool     align_msb,
                   bool     conjugate) {
	// ..................ABCDEFGH......
	// ......ABCD............EFGH......
	// AB......CD......EF......GH......
	oval = ival << 6;
	oval = (oval | (oval << 12)) & 0x03C003C0;
	oval = (oval | (oval <<  6)) & 0xC0C0C0C0;
	if( byte_reverse) {
		byteswap(oval, &oval);
	}
	if( !align_msb ) {
		// >>>>>>AB>>>>>>CD>>>>>>EF>>>>>>GH
		rshift_subwords<6,int8_t>(oval);
	}
	if( conjugate ) {
		conjugate_subwords<int8_t>(oval);
	}
}
// 8x 1-bit --> 8x 8-bit (signed)
inline void unpack(uint8_t  ival,
                   int64_t& oval,
                   bool     byte_reverse,
                   bool     align_msb,
                   bool     conjugate) {
	// .................................................ABCDEFGH.......
	// .....................ABCD............................EFGH.......
	// .......AB..............CD..............EF..............GH.......
	// A.......B.......C.......D.......E.......F.......G.......H.......
	oval = ival << 7;
	oval = (oval | (oval << 28)) & 0x0000078000000780;
	oval = (oval | (oval << 14)) & 0x0180018001800180;
	oval = (oval | (oval <<  7)) & 0x8080808080808080;
	if( byte_reverse) {
		byteswap(oval, &oval);
	}
	if( !align_msb ) {
		// >>>>>>>A>>>>>>>B>>>>>>>C>>>>>>>D
		rshift_subwords<7,int8_t>(oval);
	}
	if( conjugate ) {
		conjugate_subwords<int8_t>(oval);
	}
}

template<typename IType, typename OType>
struct UnpackFunctor {
	bool byte_reverse;
	bool align_msb;
	bool conjugate;
	UnpackFunctor(bool byte_reverse_,
	              bool align_msb_,
	              bool conjugate_)
		: byte_reverse(byte_reverse_),
		  align_msb(align_msb_),
		  conjugate(conjugate_) {}
	void operator()(IType ival, OType& oval) const {
		unpack(ival, oval, byte_reverse, align_msb, conjugate);
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

BFstatus bfUnpack(BFarray const* in,
                  BFarray const* out,
                  BFbool         align_msb) {
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
	
	// **TODO: This does not correctly interpret strides when itemsize_nbit < 8
	//           Need to work out how to handle strides when itemsize_bits < 8
	//             Perhaps some way of specifying packed contiguous inner dims
	//             Whatever solution is chosen will need to be integrated into
	//               bf.ndarray and bf.DataType.
	
	// TODO: Support padded arrays
	BF_ASSERT(is_contiguous(in),  BF_STATUS_UNSUPPORTED_STRIDE);
	BF_ASSERT(is_contiguous(out), BF_STATUS_UNSUPPORTED_STRIDE);
	
	// TODO: Support CUDA space
	BF_ASSERT(space_accessible_from(in->space, BF_SPACE_SYSTEM),
	          BF_STATUS_UNSUPPORTED_SPACE);
	BF_ASSERT(space_accessible_from(out->space, BF_SPACE_SYSTEM),
	          BF_STATUS_UNSUPPORTED_SPACE);
	
	size_t nelement = num_contiguous_elements(in);
	bool byteswap    = ( in->big_endian != is_big_endian());
	bool conjugate   = (in->conjugated != out->conjugated);
	
#define CALL_FOREACH_SIMPLE_CPU_UNPACK(itype,otype) \
	foreach_simple_cpu((itype*)in->data, \
	                   (otype*)out->data, \
	                   nelement, \
	                   UnpackFunctor<itype,otype>(byteswap, \
	                                              align_msb, \
	                                              conjugate))
	if( out->dtype == BF_DTYPE_I8 ||
	           out->dtype == BF_DTYPE_CI8 ) {
	//case BF_DTYPE_I8: {
		switch( in->dtype ) {
		// TODO: Work out how to properly deal with 1-bit
		//case BF_DTYPE_CI1: nelement *= 2;
		//case BF_DTYPE_I1: {
		//	BF_ASSERT(nelement % 8 == 0, BF_STATUS_INVALID_SHAPE);
		//	nelement /= 8;
		//	CALL_FOREACH_SIMPLE_CPU_UNPACK(uint8_t,int64_t); break;
		//}
		case BF_DTYPE_CI2: nelement *= 2;
		case BF_DTYPE_I2: {
			BF_ASSERT(nelement % 4 == 0, BF_STATUS_INVALID_SHAPE);
			nelement /= 4;
			CALL_FOREACH_SIMPLE_CPU_UNPACK(uint8_t,int32_t);
			break;
		}
		case BF_DTYPE_CI4: nelement *= 2;
		case BF_DTYPE_I4: {
			BF_ASSERT(nelement % 2 == 0, BF_STATUS_INVALID_SHAPE);
			nelement /= 2;
			CALL_FOREACH_SIMPLE_CPU_UNPACK(uint8_t,int16_t);
			break;
		}
		//case BF_DTYPE_U1: {
		//	// TODO
		//}
		case BF_DTYPE_U2: {
			BF_ASSERT(nelement % 4 == 0, BF_STATUS_INVALID_SHAPE);
			nelement /= 4;
			CALL_FOREACH_SIMPLE_CPU_UNPACK(uint8_t,uint32_t);
			break;
		}
		case BF_DTYPE_U4: {
			BF_ASSERT(nelement % 2 == 0, BF_STATUS_INVALID_SHAPE);
			nelement /= 2;
			CALL_FOREACH_SIMPLE_CPU_UNPACK(uint8_t,uint16_t);
			break;
		}
		default: BF_FAIL("Supported bfQuantize input dtype", BF_STATUS_UNSUPPORTED_DTYPE);
		}
	} else {
		BF_FAIL("Supported bfQuantize output dtype", BF_STATUS_UNSUPPORTED_DTYPE);
	}
#undef CALL_FOREACH_SIMPLE_CPU_UNPACK
	return BF_STATUS_SUCCESS;
}
