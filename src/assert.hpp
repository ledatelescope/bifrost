/*
 *  Copyright 2016 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#pragma once

#include <bifrost/common.h>

#include <stdexcept>

class BFexception : public std::runtime_error {
	BFstatus _status;
public:
	BFexception(BFstatus stat, const char* msg="")
		: std::runtime_error(std::string(bfGetStatusString(stat))+
		                     ": "+msg),
		  _status(stat) {}
	BFstatus status() const { return _status; }
};

#if defined(BF_DEBUG) && BF_DEBUG
#include <iostream>
using std::cout;
using std::endl;
#define BF_REPORT_ERROR(err) do { \
		std::cerr << __FILE__ << ":" << __LINE__ \
		          << " error " << err << ": " \
		          << bfGetStatusString(err) << std::endl; \
	} while(0)
#define BF_REPORT_INTERNAL_ERROR(msg) do { \
		std::cerr << __FILE__ << ":" << __LINE__ \
		          << " internal error: " \
		          << msg << std::endl; \
	} while(0)
#define BF_DEBUG_PRINT(x) \
	std::cout << __FILE__ << ":" << __LINE__ \
	<< " " #x << "\t=\t" << (x) << std::endl
#else
#define BF_REPORT_ERROR(err)
#define BF_DEBUG_PRINT(x)
#endif // BF_DEBUG

#define BF_ASSERT(pred, err) do { \
		if( !(pred) ) { \
			BF_REPORT_ERROR(err); \
			return (err); \
		} \
	} while(0)
#define BF_TRY(code, onfail) do { \
		try { code; return BF_STATUS_SUCCESS; } \
		catch( BFexception const& err ) { \
			onfail; \
			if( err.status() != BF_STATUS_END_OF_DATA ) { \
				BF_REPORT_ERROR(err.status()); \
			} \
			return err.status(); \
		} \
		catch(std::exception const& err) { \
			onfail; \
			BF_REPORT_INTERNAL_ERROR(err.what()); \
			return BF_STATUS_INTERNAL_ERROR; \
		} \
		catch(...) { \
			onfail; \
			BF_REPORT_INTERNAL_ERROR("FOREIGN EXCEPTION"); \
			return BF_STATUS_INTERNAL_ERROR; \
		} \
	} while(0)

#define BF_ASSERT_EXCEPTION(pred, err) \
	do { \
		if( !(pred) ) { \
			/*if( err != BF_STATUS_END_OF_DATA )*/ { \
				BF_REPORT_ERROR(err); \
			} \
			throw BFexception(err); \
		} \
	} while(0)

#define BF_NO_OP (void)0
