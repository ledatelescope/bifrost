/*
 * Copyright (c) 2016-2022, The Bifrost Authors. All rights reserved.
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

/*
bfMap(3, c.shape, {"dm", "t"},
      {a, b, c}, {"a", "b", "c"},
      "c(dm,t) = a(dm,t) + a(dm,t/2);" // Explicit indexing
      "auto ab = a(dm,t) + b(dm,t);" // Use of temporaries
      "c(dm,t) = ab*conj(ab);" // Built-in math functions
      "c(_) = a(_) + b(_)"); // '_' is the full composite index
*/

// This enables print-out of generated source and PTX
//#define BF_DEBUG_RTC 1

#include <bifrost/config.h>
#include <bifrost/map.h>
#include "fileutils.hpp"

#include "cuda.hpp"
#include "utils.hpp"
#include "assert.hpp"
#include "array_utils.hpp"
#include "ObjectCache.hpp"
#include "EnvVars.hpp"

#include <cuda.h>
#include <nvrtc.h>

#include "IndexArray.cuh.jit"
#include "ArrayIndexer.cuh"
#include "ArrayIndexer.cuh.jit"
#include "ShapeIndexer.cuh"
#include "ShapeIndexer.cuh.jit"
#include "Complex.hpp"
#include "Complex.hpp.jit"
#include "Vector.hpp"
#include "Vector.hpp.jit"
#include "int_fastdiv.h.jit"

#include <vector>
#include <sstream>
#include <iomanip>
#include <unordered_map>

#include <iostream>
using std::cout;
using std::cerr;
using std::endl;

#include <fstream>
#include <set>
#include <mutex>

#include <sys/types.h>
#include <dirent.h>

#define BF_CHECK_NVRTC(call) \
	do { \
		nvrtcResult ret = call; \
		if( ret != NVRTC_SUCCESS ) { \
			BF_DEBUG_PRINT(nvrtcGetErrorString(ret)); \
		} \
		BF_ASSERT(ret == NVRTC_SUCCESS, \
		          bifrost_status(ret)); \
	} while(0)

BFstatus bifrost_status(nvrtcResult status) {
	switch(status) {
	case NVRTC_SUCCESS:                         return BF_STATUS_SUCCESS;
	case NVRTC_ERROR_OUT_OF_MEMORY:             return BF_STATUS_MEM_ALLOC_FAILED;
	case NVRTC_ERROR_PROGRAM_CREATION_FAILURE:  return BF_STATUS_INTERNAL_ERROR;
	case NVRTC_ERROR_INVALID_INPUT:             return BF_STATUS_INTERNAL_ERROR;
	case NVRTC_ERROR_INVALID_PROGRAM:           return BF_STATUS_INTERNAL_ERROR;
	case NVRTC_ERROR_INVALID_OPTION:            return BF_STATUS_INTERNAL_ERROR;
	case NVRTC_ERROR_COMPILATION:               return BF_STATUS_INTERNAL_ERROR;
	case NVRTC_ERROR_BUILTIN_OPERATION_FAILURE: return BF_STATUS_INTERNAL_ERROR;
#if CUDA_VERSION >= 7500
	case NVRTC_ERROR_INTERNAL_ERROR:            return BF_STATUS_DEVICE_ERROR;
#endif
	default: return BF_STATUS_INTERNAL_ERROR;
    }
}

BFstatus build_map_kernel(int*                 external_ndim,
                          long*                external_shape,
                          long                 block_x_axis,
                          long                 block_y_axis,
                          char const*const*    axis_names,
                          int                  narg,
                          BFarray const*const* args,
                          char const*const*    arg_names,
                          char const*          func_name,
                          char const*          func,
                          char const*          extra_code,
                          bool                 basic_indexing_only,
                          std::string*         ptx_string,
                          std::string*         kernel_name_ptr) {
	// Make local copies of ndim and shape to avoid corrupting external copies
	//   until we know that this function has succeeded.
	// TODO: This is not very elegant
	int ndim = *external_ndim;
	long shape[BF_MAX_DIMS];
	::memcpy(shape, external_shape, ndim*sizeof(*shape));
	long y_size = 1;
	long x_size = 1;
	
	std::vector<BFarray>  mutable_arrays;
	std::vector<BFarray*> mutable_array_ptrs;
	if( basic_indexing_only ) {
		unsigned long keep_dims_mask = 0;
		for( int a=0; a<narg; ++a ) {
			// Must not flatten padded or broadcast dims
			// Note: dim_delta accounts for tail-aligned broadcasting
			int dim_delta = ndim - args[a]->ndim;
			keep_dims_mask |= padded_dims_mask(args[a]) << dim_delta;
			keep_dims_mask |= broadcast_dims_mask(args[a], ndim, shape);
		}
		flatten_shape(&ndim, shape, keep_dims_mask);
		mutable_arrays.resize(narg);
		mutable_array_ptrs.resize(narg);
		for( int a=0; a<narg; ++a ) {
			mutable_array_ptrs[a] = &mutable_arrays[a];
			flatten(args[a], mutable_array_ptrs[a], keep_dims_mask);
		}
		args = &mutable_array_ptrs[0];
	} else {
		// TODO: This code is duplicated in the main bfMap function below
		if( ndim >= 2 ) {
			y_size = shape[block_y_axis];
			shape[block_y_axis] = 1;
		}
		if( ndim >= 1 ) {
			x_size = shape[block_x_axis];
			shape[block_x_axis] = 1;
		}
	}
	
	std::string kernel_name;
	if( func_name ) {
		kernel_name = func_name;
		kernel_name += "_";
	}
	kernel_name += "map_kernel";
	std::stringstream code;
	code << "#include \"Complex.hpp\"" << endl;
	code << "#include \"Vector.hpp\"" << endl;
	code << "#include \"ArrayIndexer.cuh\"" << endl;
	code << "#include \"ShapeIndexer.cuh\"" << endl;
	if( extra_code ) {
		code << "\n" << extra_code << "\n" << endl;
	}
	code << "extern \"C\"\n";
	code << "__global__\n";
	code << "void " << kernel_name << "(";
	for( int a=0; a<narg; ++a ) {
		std::string ctype_string = dtype2ctype_string(args[a]->dtype);
		BF_ASSERT(ctype_string.size(), BF_STATUS_INVALID_ARGUMENT);
		if( args[a]->ndim     == 1 &&
		    args[a]->shape[0] == 1 &&
		    args[a]->immutable &&
		    space_accessible_from(args[a]->space, BF_SPACE_SYSTEM) ) {
			// Special case for scalar parameters
			code << "  " << ctype_string
			     << " const"
			     << " " << arg_names[a];
				
		} else {
			code << ctype_string
			     << (args[a]->immutable ? " const" : "")
			     << "* " << arg_names[a] << "_ptr";
		}
		if( a != narg-1 ) {
			code << ",\n";
		}
	}
	code << ") {\n";
	code << "  enum { NDIM = " << ndim << " };\n";
	code << "  typedef StaticIndexArray<int,";
	for( int d=0; d<ndim; ++d ) {
		code << shape[d] << (d!=ndim-1 ? "," : "");
	}
	code << "> _Shape;\n";
	code << "  typedef StaticShapeIndexer<_Shape> _ShapeIndexer;\n";
	for( int a=0; a<narg; ++a ) {
		std::string ctype_string = dtype2ctype_string(args[a]->dtype);
		code << "  typedef StaticIndexArray<int,";
		for( int d=0; d<args[a]->ndim; ++d ) {
			code << args[a]->shape[d] << (d!=args[a]->ndim-1 ? "," : "");
		}
		code << "> _Shape_" << arg_names[a] << ";\n";
		code << "  typedef StaticIndexArray<int,";
		for( int d=0; d<args[a]->ndim; ++d ) {
			code << args[a]->strides[d] << (d!=args[a]->ndim-1 ? "," : "");
		}
		code << "> _Strides_" << arg_names[a] << ";\n";
		code << (basic_indexing_only ?
		         "  typedef StaticArrayIndexerBasic<" :
		         "  typedef StaticArrayIndexer<")
		     << ctype_string << (args[a]->immutable ? " const" : "")
		     << ",_Shape_"         << arg_names[a]
		     << ",_Strides_"       << arg_names[a]
		     << "> _ArrayIndexer_" << arg_names[a] << ";\n";
	}
	// Add a _shape array for the user to use if needed
	code << "  const int _shape[NDIM] = {";
	for( int d=0; d<ndim; ++d ) {
		if( d == block_x_axis && !basic_indexing_only ) {
			code << x_size;
		} else if( d == block_y_axis && !basic_indexing_only ) {
			code << y_size;
		} else {
			code << shape[d];
		}
		if( d != ndim-1 ) {
			code << ", ";
		}
	}
	code << "}; (void)_shape[0];\n"; // Note: Prevents unused variable warning
	if( basic_indexing_only ) {
		code <<
			"  int _x0 = threadIdx.x + blockIdx.x*blockDim.x;\n"
			"  for( int _x=_x0; _x<_ShapeIndexer::SIZE; _x+=blockDim.x*gridDim.x ) {\n"
			"    auto const& _  = _ShapeIndexer::lift(_x);\n";
	} else {
		code <<
			"  int _x0 = threadIdx.x + blockIdx.x*blockDim.x;\n";
		if( y_size > 1 ) {
			code << "  int _y0 = threadIdx.y + blockIdx.y*blockDim.y;\n";
		}
		code <<
			"  int _z0 = blockIdx.z;\n"
			"  for( int _z=_z0; _z<_ShapeIndexer::SIZE; _z+=gridDim.z ) {\n";
			// TODO: Condition this out if y_size == 1
		if( y_size > 1 ) {
			code << "  for( int _y=_y0; _y<" << y_size << "; _y+=blockDim.y*gridDim.y ) {\n";
		}
		code <<
			"  for( int _x=_x0; _x<" << x_size << "; _x+=blockDim.x*gridDim.x ) {\n"
			"    auto _composite_index  = _ShapeIndexer::lift(_z);\n"
			"    _composite_index[" << block_x_axis << "] = _x;\n";
		if( y_size > 1 ) {
			code <<
				"    _composite_index[" << block_y_axis << "] = _y;\n";
		}
		// Create the composite index variable for use by the user
		code << "    auto const& _  = _composite_index;\n";
	}
	for( int a=0; a<narg; ++a ) {
		if( args[a]->ndim     == 1 &&
		    args[a]->shape[0] == 1 &&
		    args[a]->immutable &&
		    space_accessible_from(args[a]->space, BF_SPACE_SYSTEM) ) {
			// pass
		} else {
			// TODO: De-dupe this with the one above
			std::string ctype_string = dtype2ctype_string(args[a]->dtype);
			if( basic_indexing_only ) {
				// Here we define the variable as a plain reference
				code << "    _ArrayIndexer_" << arg_names[a] << " "
				     << "__" << arg_names[a] << "(" << arg_names[a] << "_ptr, _);\n";
				code << "    auto& " << arg_names[a] << " = *__" << arg_names[a] << ";\n";
			} else {
				// Here we define the variable as a StaticArrayIndexer instance
				code << "    _ArrayIndexer_" << arg_names[a] << " "
				     << arg_names[a] << "(" << arg_names[a] << "_ptr, _);\n";
			}
			code << "    typedef " << ctype_string << " " << arg_names[a] << "_type;\n";
		}
	}
	// Create named axis variables
	if( axis_names ) {
		for( int d=0; d<ndim; ++d ) {
			BF_ASSERT(axis_names[d][0] != '_', BF_STATUS_INVALID_ARGUMENT);
			code << "    auto " << axis_names[d] << " = _[" << d << "];\n";
		}
	}
	code << "    " << func << ";\n";
	code << "  }\n"; // End _x loop
	if( !basic_indexing_only ) {
		if( y_size > 1 ) {
			code << "  }\n"; // End _y loop
		}
		code << "  }\n"; // End _z loop
	}
	code << "}\n"; // End kernel
	
	const char* program_name = func_name ? func_name : "bfMap";
	const char* header_codes[] = {
		Complex_hpp,
		Vector_hpp,
		ArrayIndexer_cuh,
		ShapeIndexer_cuh,
		IndexArray_cuh,
		int_fastdiv_h
	};
	const char* header_names[] = {
		"Complex.hpp",
		"Vector.hpp",
		"ArrayIndexer.cuh",
		"ShapeIndexer.cuh",
		"IndexArray.cuh",
		"int_fastdiv.h" // TODO: Don't actually need this, it's just an unused depdency of ShapeIndexer.cuh; try to remove it
	};
	size_t nheader = sizeof(header_codes) / sizeof(const char*);
	
	nvrtcProgram program;
	BF_CHECK_NVRTC( nvrtcCreateProgram(&program,
	                                   code.str().c_str(),
	                                   program_name,
	                                   nheader, header_codes, header_names) );
	std::vector<std::string> options;
	std::stringstream cs_ss;
#if defined(BF_MAP_KERNEL_STDCXX)
	cs_ss << BF_MAP_KERNEL_STDCXX;
#else
	cs_ss << "c++11";
#endif
	options.push_back("--std="+cs_ss.str());
	options.push_back("--device-as-default-execution-space");
	options.push_back("--use_fast_math");
	std::stringstream cc_ss;
#if defined(BF_MAP_KERNEL_DISK_CACHE) && BF_MAP_KERNEL_DISK_CACHE
	cc_ss << "compute_" << BF_GPU_MIN_ARCH;
#else
	cc_ss << "compute_" << get_cuda_device_cc();
#endif
	options.push_back("-arch="+cc_ss.str());
	options.push_back("--restrict");
	std::vector<const char*> options_c;
	for( int i=0; i<(int)options.size(); ++i ) {
		options_c.push_back(options[i].c_str());
	}
	nvrtcResult ret = nvrtcCompileProgram(program,
	                                      options_c.size(),
	                                      &options_c[0]);
#if BF_DEBUG_ENABLED
	size_t logsize;
	// Note: Includes the trailing NULL
	BF_CHECK_NVRTC( nvrtcGetProgramLogSize(program, &logsize) );
	if( (logsize > 1 || EnvVars::get("BF_PRINT_MAP_KERNELS", "0") != "0") &&
	     !basic_indexing_only ) {
		std::vector<char> log(logsize, 0);
		BF_CHECK_NVRTC( nvrtcGetProgramLog(program, &log[0]) );
		int i = 1;
		for( std::string line; std::getline(code, line); ++i ) {
			std::cout << std::setfill(' ') << std::setw(3) << i << " " << line << endl;
		}
		std::cout << "---------------------------------------------------" << std::endl;
		std::cout << "--- JIT compile log for program " << program_name << " ---" << std::endl;
		std::cout << "---------------------------------------------------" << std::endl;
		std::cout << &log[0] << std::endl;
		std::cout << "---------------------------------------------------" << std::endl;
	}
#endif // BIFROST_DEBUG
	if( ret != NVRTC_SUCCESS ) {
		// Note: Don't print debug msg here, failure may not be expected
		return BF_STATUS_INVALID_ARGUMENT;
	}
	
	size_t ptxsize;
	BF_CHECK_NVRTC( nvrtcGetPTXSize(program, &ptxsize) );
	std::vector<char> vptx(ptxsize);
	char* ptx = &vptx[0];
	BF_CHECK_NVRTC( nvrtcGetPTX(program, &ptx[0]) );
	BF_CHECK_NVRTC( nvrtcDestroyProgram(&program) );
#if BF_DEBUG_ENABLED
	if( EnvVars::get("BF_PRINT_MAP_KERNELS_PTX", "0") != "0" ) {
		std::cout << ptx << std::endl;
	}
#endif
	*ptx_string = ptx;
	*kernel_name_ptr = kernel_name;
	// TODO: Can't do this, because this function may be cached
	//         **Clean this up!
	//*external_ndim = ndim;
	//*external_y_size = y_size;
	//*external_x_size = x_size;
	//::memcpy(external_shape, shape, ndim*sizeof(*shape));
	return BF_STATUS_SUCCESS;
}

#if defined(BF_MAP_KERNEL_DISK_CACHE) && BF_MAP_KERNEL_DISK_CACHE
class DiskCacheMgr {
	std::string            _cachedir;
	std::string            _indexfile;
	std::string            _cachefile;
	std::set<std::string>  _created_dirs;
	mutable std::mutex     _mutex;
	bool                   _loaded;
	std::hash<std::string> _get_name;
	
	void tag_cache(void) {
		// NOTE:  Must be called from within a LockFile lock
		int rt, drv;
		cudaRuntimeGetVersion(&rt);
		cudaDriverGetVersion(&drv);

		std::ofstream info;
		
		if( !file_exists(_cachedir + BF_MAP_KERNEL_DISK_CACHE_VERSION_FILE) ) {
		    try {
		        info.open(_cachedir + BF_MAP_KERNEL_DISK_CACHE_VERSION_FILE, std::ios::out);
		        info << BF_MAP_KERNEL_DISK_CACHE_VERSION << " " << rt << " " << drv << endl;
		        info.close();
		    } catch( std::exception const& ) {}
		}
	}
	
	void validate_cache(void) {
		// NOTE:  Must be called from within a LockFile lock
		bool status = true;
		int rt, drv, cached_mc, cached_rt, cached_drv;
		cudaRuntimeGetVersion(&rt);
		cudaDriverGetVersion(&drv);
		
		std::ifstream info;
		try {
		    // Open
		    info.open(_cachedir + BF_MAP_KERNEL_DISK_CACHE_VERSION_FILE, std::ios::in);
		    
		    // Read
		    info >> cached_mc >> cached_rt >> cached_drv;
		    
		    // Close
		    info.close();
		} catch( std::exception const& ) {
		    cached_mc = cached_rt = cached_drv = -1;
		}
		
		if( BF_MAP_KERNEL_DISK_CACHE_VERSION != cached_mc || rt != cached_rt || drv != cached_drv ) {
		    status = false;
		}
		
		if( !status ) {
		    try {
          remove_files_with_suffix(_cachedir, ".inf");
          remove_files_with_suffix(_cachedir, ".ptx");
          remove_file(_cachedir + BF_MAP_KERNEL_DISK_CACHE_VERSION_FILE);
	        } catch( std::exception const& ) {}
	    }
	}
	
	// TODO:  Per arch. caches/files?  Compile to cubin and cache per arch.?
	void write_to_disk(std::string  cache_key, 
                     std::string  kernel_name,
                     std::string  ptx,  
                     bool         basic_indexing_only) {              
        // Do this with a file lock to avoid interference from other processes
				LockFile lock(_cachedir + ".lock");
		
        this->tag_cache();
        
        // Get the name to save the kernel to
        std::stringstream basename;
        basename << std::hex << std::uppercase << _get_name(cache_key);
        _indexfile = _cachedir + basename.str() + ".inf";
        _cachefile = _cachedir + basename.str() + ".ptx";
        
        std::ofstream index, cache;
        try {
            // Open
            index.open(_indexfile, std::ios::out);
            cache.open(_cachefile, std::ios::out|std::ios::binary);
            
            // Write Info
            //  * PTX size
            index << ptx.size() << endl;
            //  * Kernel name
            index << kernel_name << endl;
            //  * Whether or not the kernel supports basic indexing only
            index << basic_indexing_only << endl;
            //  * In-memory cache key
            index << cache_key;
            
            // Write PTX
            cache.write(ptx.c_str(), ptx.size());
	        
	        // Done
	        index.close();
	        cache.close();
	    } catch( std::exception const& ) {}
	}
	
	void load_from_disk(ObjectCache<std::string,std::pair<CUDAKernel,bool> > *kernel_cache) {
	    // Do this with a file lock to avoid interference from other processes
			LockFile lock(_cachedir + ".lock");

      // Validate the cache
			this->validate_cache();

    	std::ifstream index, cache;
      size_t kernel_size;
      std::string kernel_name;
      bool basic_indexing_only;
      std::string cache_key;
      std::string ptx;
      CUDAKernel kernel;
      
      std::string field;
      
      // Find the files
      DIR* dir = opendir(_cachedir.c_str());
      struct dirent *entry;
      while( (entry = readdir(dir)) != NULL ) {
      		_indexfile = _cachedir + std::string(entry->d_name);
          if( _indexfile.size() < 4 ) {
              continue;
          }
          if( _indexfile.compare(_indexfile.size()-4, 4, ".inf") == 0 ) {
              // We have an info file, get the corresponding ptx file
              _cachefile = _indexfile.substr(0, _indexfile.size()-4) + ".ptx";
              
              try {
                  // Open
                  index.open(_indexfile, std::ios::in);
                  cache.open(_cachefile, std::ios::in|std::ios::binary);
                      
                  // Read in info and then build the kernels
                  //  * PTX size
                  std::getline(index, field);
                  kernel_size = std::atoi(field.c_str());
                  //  * Kernel name
                  std::getline(index, kernel_name);
                  //  * Whether or not the kernel supports basic indexing only
                  std::getline(index, field);
                  basic_indexing_only = false;
                  if( std::atoi(field.c_str()) > 0 ) {
                      basic_indexing_only = true;
                  }
                  //  * In-memory cache name
                  std::getline(index, cache_key);
                  while( std::getline(index, field) ) {
                      cache_key = cache_key + "\n" + field;
                  }
                  
                  // Check the cache to see if it is aleady there...
                  if( !kernel_cache->contains(cache_key) ) {
                      ptx.resize(kernel_size);
                      cache.read(&ptx[0], kernel_size);
                      kernel.set(kernel_name.c_str(), ptx.c_str());
                      
                      kernel_cache->insert(cache_key,
	                                         std::make_pair(kernel, basic_indexing_only));         
			            }
									
		            	// Done
	      	        index.close();
	                cache.close();
	            } catch( std::exception const&) {}
	        }
	    }
	    closedir(dir);
	}
	
	void clear_cache() {
	    // Do this with a file lock to avoid interference from other processes
		LockFile lock(_cachedir + ".lock");

    try {
        remove_files_with_suffix(_cachedir, ".inf");
        remove_files_with_suffix(_cachedir, ".ptx");
    } catch( std::exception const& ) {}
  }

	DiskCacheMgr()
		: _cachedir(get_home_dir()+"/.bifrost/"+BF_MAP_KERNEL_DISK_CACHE_SUBDIR+"/"),
		  _loaded(false) {
				make_dir(_cachedir);
	}
public:
	DiskCacheMgr(DiskCacheMgr& ) = delete;
	DiskCacheMgr& operator=(DiskCacheMgr& ) = delete;
	
	static DiskCacheMgr& get() {
		static DiskCacheMgr cache;
		return cache;
	}
	
	bool load(ObjectCache<std::string,std::pair<CUDAKernel,bool> > *kernel_cache) {
		if( !_loaded ) {
		    std::lock_guard<std::mutex> lock(_mutex);
		    this->load_from_disk(kernel_cache);
	        _loaded = true;
		}
		return _loaded;
	}
	
	void save(std::string  cache_key, 
              std::string  kernel_name,
              std::string  ptx,  
              bool         basic_indexing_only) {
         std::lock_guard<std::mutex> lock(_mutex);
         this->write_to_disk(cache_key, kernel_name, ptx, basic_indexing_only);
    }
    
    void clear() {
        std::lock_guard<std::mutex> lock(_mutex);
        this->clear_cache();
        _loaded = false;
    }
};
#endif

BFstatus bfMap(int                  ndim,
               long const*          shape,
               char const*const*    axis_names,
               int                  narg,
               BFarray const*const* args,
               char const*const*    arg_names,
               char const*          func_name,
               char const*          func,
               char const*          extra_code,
               int  const           block_shape[2],
               int  const           block_axes[2]) {
	// Map containing compiled kernels and basic_indexing_only flag
	thread_local static ObjectCache<std::string,std::pair<CUDAKernel,bool> >
	kernel_cache(BF_MAP_KERNEL_CACHE_SIZE);
	BF_ASSERT(ndim >= 0,           BF_STATUS_INVALID_ARGUMENT);
	//BF_ASSERT(!ndim || shape,      BF_STATUS_INVALID_POINTER);
	//BF_ASSERT(!ndim || axis_names, BF_STATUS_INVALID_POINTER);
	BF_ASSERT(narg >= 0,           BF_STATUS_INVALID_ARGUMENT);
	BF_ASSERT(!narg || args,       BF_STATUS_INVALID_POINTER);
	BF_ASSERT(!narg || arg_names,  BF_STATUS_INVALID_POINTER);
	BF_ASSERT(func,                BF_STATUS_INVALID_POINTER);
	//if( ndim == 0 ) {
	//	return BF_STATUS_SUCCESS;
	//}
	// TODO: Could derive these from a heuristic that looks at the fastest-changing input/output dims
	const int block_axes_default[] = {ndim-2, ndim-1};
	bool force_advanced_indexing = (shape || block_axes);
	if( !block_axes ) {
		block_axes = block_axes_default;
	}
	
	long mutable_shape[BF_MAX_DIMS];
	if( !shape ) {
		BF_ASSERT(broadcast_shapes(narg, args, mutable_shape, &ndim),
		          BF_STATUS_INVALID_SHAPE);
	} else {
		::memcpy(mutable_shape, shape, ndim*sizeof(*shape));
	}
	shape = mutable_shape;
	
	int block_x_axis = block_axes[1];
	int block_y_axis = block_axes[0];
	// Support negative indexing
	block_x_axis = (block_x_axis < 0) ? ndim + block_x_axis : block_x_axis;
	block_y_axis = (block_y_axis < 0) ? ndim + block_y_axis : block_y_axis;
	
	std::stringstream cache_key_ss;
	cache_key_ss << ndim << ",";
	for( int d=0; d<ndim; ++d ) {
		cache_key_ss << shape[d] << ",";
		const char* axis_name = axis_names ? axis_names[d] : "";
		cache_key_ss << axis_name << ",";
	}
	for( int a=0; a<narg; ++a ) {
		cache_key_ss << arg_names[a] << ","
		             << args[a]->dtype << ","
		             << args[a]->immutable << ","
		             << args[a]->space << ","
		             << args[a]->ndim << ",";
		for( int d=0; d<args[a]->ndim; ++d ) {
			cache_key_ss << args[a]->shape[d] << ",";
			cache_key_ss << args[a]->strides[d] << ",";
		}
	}
	cache_key_ss << force_advanced_indexing << ",";
	cache_key_ss << block_x_axis << ",";
	cache_key_ss << block_y_axis << ",";
	cache_key_ss << func << ",";
	if( extra_code ) {
		cache_key_ss << extra_code;
	}
	std::string cache_key = cache_key_ss.str();
	
#if defined(BF_MAP_KERNEL_DISK_CACHE) && BF_MAP_KERNEL_DISK_CACHE
    DiskCacheMgr::get().load(&kernel_cache);
#endif
	
	if( !kernel_cache.contains(cache_key) ) {
		std::string ptx;
		std::string kernel_name;
		// First we try with basic_indexing_only = true
		bool basic_indexing_only = true;
		if( force_advanced_indexing ||
		    build_map_kernel(&ndim, mutable_shape,
		                     block_x_axis, block_y_axis,
		                     axis_names, narg,
		                     args, arg_names,
		                     func_name, func, extra_code,
		                     basic_indexing_only,
		                     &ptx, &kernel_name) != BF_STATUS_SUCCESS ) {
			// Then we fall back to basic_indexing_only = false
			basic_indexing_only = false;
			BF_CHECK(build_map_kernel(&ndim, mutable_shape,
			                          block_x_axis, block_y_axis,
			                          axis_names, narg,
			                          args, arg_names,
			                          func_name, func, extra_code,
			                          basic_indexing_only,
			                          &ptx, &kernel_name));
		}
		CUDAKernel kernel;
		BF_TRY(kernel.set(kernel_name.c_str(), ptx.c_str()));
		kernel_cache.insert(cache_key,
		                    std::make_pair(kernel, basic_indexing_only));
#if defined(BF_MAP_KERNEL_DISK_CACHE) && BF_MAP_KERNEL_DISK_CACHE
		DiskCacheMgr::get().save(cache_key, kernel_name, ptx, basic_indexing_only);
#endif
	}
	auto& cache_entry = kernel_cache.get(cache_key);
	CUDAKernel& kernel = cache_entry.first;
	bool basic_indexing_only = cache_entry.second;
	
	int x_size = 1, y_size = 1;
	if( !basic_indexing_only ) {
		if( ndim >= 2 ) {
			y_size = shape[block_y_axis];
			mutable_shape[block_y_axis] = 1;
		}
		if( ndim >= 1 ) {
			x_size = shape[block_x_axis];
			mutable_shape[block_x_axis] = 1;
		}
	}
	
	std::vector<void*> kernel_args;
	kernel_args.reserve(narg);
	for( int a=0; a<narg; ++a ) {
		if( args[a]->ndim     == 1 &&
		    args[a]->shape[0] == 1 &&
		    args[a]->immutable &&
		    space_accessible_from(args[a]->space, BF_SPACE_SYSTEM) ) {
			// Special case for scalar parameters
			kernel_args.push_back(args[a]->data);
		} else {
			BF_ASSERT(args[a]->data, BF_STATUS_INVALID_POINTER);
			BF_ASSERT(space_accessible_from(args[a]->space, BF_SPACE_CUDA),
			          BF_STATUS_INVALID_SPACE);
			
			kernel_args.push_back((void**)&args[a]->data);
		}
	}
	
	long nelement = shape_size(ndim, shape);
	const int block_shape_default[] = {16, 32};
	if( !block_shape ) {
		block_shape = block_shape_default;
	}
	dim3 block, grid;
	if( basic_indexing_only || ndim < 2 ) {
		block = dim3(block_shape[1]*block_shape[0]);
	} else {
		block = dim3(block_shape[1], block_shape[0]);
	}
	if( basic_indexing_only ) {
		grid = dim3(std::min((nelement-1)/block.x+1, 65535l));
	} else {
		grid = dim3(std::min((x_size-1)/block.x+1, 65535u),
		            std::min((y_size-1)/block.y+1, 65535u),
		            std::min(nelement, 65535l));
	}
	
	BF_ASSERT(kernel.launch(grid, block,
	                        0, g_cuda_stream,
	                        kernel_args) == CUDA_SUCCESS,
	          BF_STATUS_DEVICE_ERROR);
	
	return BF_STATUS_SUCCESS;
}

BFstatus bfMapClearCache() {
#if defined(BF_MAP_KERNEL_DISK_CACHE) && BF_MAP_KERNEL_DISK_CACHE
    DiskCacheMgr::get().clear();
#endif
    return BF_STATUS_SUCCESS;
}
