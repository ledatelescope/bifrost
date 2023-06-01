
/*
 * Copyright (c) 2021, The Bifrost Authors. All rights reserved.
 * Copyright (c) 2021, The University of New Mexico. All rights reserved.
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

/*! \file pypy3_compat.c
 *  \brief Compatibility layer for PyPy3
 */
 
#include "Python.h"
#include <stdio.h>

static PyObject* PyMemoryView_FromAddressAndSize(PyObject *self, PyObject *args, PyObject *kwds) {
    PyObject *address, *nbyte, *flags, *view;
    if(!PyArg_ParseTuple(args, "OOO", &address, &nbyte, &flags)) {
        PyErr_Format(PyExc_RuntimeError, "Invalid parameters");
        return NULL;
    }
    
    long addr, size, flgs;
    addr = PyLong_AsLong(address);
    size = PyLong_AsLong(nbyte);
    flgs = PyLong_AsLong(flags);
    
    char *buf = (char *) addr;
    
    view = PyMemoryView_FromMemory(buf, size, flgs | PyBUF_READ);
    return view;
}

static PyMethodDef CompatMethods[] = {
  {"PyMemoryView_FromMemory", (PyCFunction) PyMemoryView_FromAddressAndSize, METH_VARARGS, NULL},
  {NULL,                      NULL,                                          0,            NULL}
};

static struct PyModuleDef Compat = {
  PyModuleDef_HEAD_INIT, "_pypy3_compat", NULL, -1, CompatMethods,};

PyMODINIT_FUNC PyInit__pypy3_compat(void) {
  PyObject *m;
  m = PyModule_Create(&Compat);
  if(m == NULL) {
    return NULL;
  }
  return m;
}
