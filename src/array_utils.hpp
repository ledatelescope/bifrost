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

#pragma once

//const char* dtype2ctype_string(BFdtype dtype);
inline const char* dtype2ctype_string(BFdtype dtype) {
	switch( dtype ) {
	case BF_DTYPE_I8:    return "signed char";
	case BF_DTYPE_I16:   return "short";
	case BF_DTYPE_I32:   return "int";
	case BF_DTYPE_I64:   return "long long";
	case BF_DTYPE_U8:    return "unsigned char";
	case BF_DTYPE_U16:   return "unsigned short";
	case BF_DTYPE_U32:   return "unsigned int";
	case BF_DTYPE_U64:   return "unsigned long long";
	case BF_DTYPE_F32:   return "float";
	case BF_DTYPE_F64:   return "double";
	case BF_DTYPE_F128:  return "long double";
	case BF_DTYPE_CI8:   return "Complex<signed char>";//complex<signed char>";
	case BF_DTYPE_CI16:  return "Complex<short>";//complex<short>";
	//case BF_DTYPE_CI32:  return "complex<int>";
	//case BF_DTYPE_CI64:  return "complex<long long>";
	case BF_DTYPE_CF32:  return "Complex<float>";//complex<float>";
	case BF_DTYPE_CF64:  return "Complex<double>";
	//case BF_DTYPE_CF128: return "complex<long double>";
	default: return 0;
	}
}
