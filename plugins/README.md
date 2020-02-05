# Bifrost Plugins

## Overview
This directory provides a collection of tools for building plugins that extend
the functionality of Bifrost.  These plugins can be seen as a way to augment
the capabilities avaliable with `bifrost.map()` and provide users a way to 
optimize their code.

## Usage and Installation
The main script for building a plugin is called `make_bifrost_plugin.py`.  This
script takes in a C++ or CUDA source file and uses that to build the shared 
object library and the Python wrappers.  To build a plugin for a source file 
called `simple_functions.cpp`:

```make_bifrost_plugin.py simple_functions.cpp
```

This will generate a `libsimple_functions.so` object and two Python scripts:
`simple_functions_generated.py` and `simple_functions.py`.  `simple_functions_gnerated.py`
is the low-level ctypesgen wrapper for the plugin and `simple_functions.py` is 
the high-level interface that can be used inside of a pipeline.

No installation of the plugin is strictly necessary to use it once it has been 
built but the Python scripts and the library.  However, you may need to update 
both the `PYTHONPATH` and `LD_LIBRARY_PATH` environment variables if you have 
import errors when trying to use a plugin.

## Examples
The plugin framework supports two different classes of plugins:
collections of functions and simple class-based operations.  Examples of both 
types are included in the `examples` directory.

### Collections of Functions
The `simple_functions.cpp` file provide an example of a plugin that implements 
simple mathematical operations that work on data stored in `BFarray`s.  The
functions `AddStuff()` and `SubtractStuff()` each take in three arrays:  two 
inputs and one output.  When the plugin is wrapped to build the Python interface
these two functions are mapped to the `add_stuff()` and `subtract_stuff()`
functions.  The function name mapping is case sensitive and care should be takes 
to match the naming conventions.

### A Simple Class
The `simple_class.cpp` file provides an example of a simple mathematical 
operation uses a class to maintain state.  The class is called `simpleclass_impl`
and the main functions inside this file are: `SimpleClassCreate()`, 
`SimpleClassInit()`, `SimpleClassExecute()`, and `SimpleClassDestroy()`.  These 
four functions serve as intermediaries for working with the underlying 
`simpleclass_impl` instance.  When the plugin is wrapped the functions are 
contained in a Python class called `SimpleClass`.  Of the four functions only
`SimpleClassInit()` and `SimpleClassExecute` are exposed in this class.
