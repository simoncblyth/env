#!/usr/bin/env python
"""
setup.py file for SWIG example

    python setup.py build_ext --inplace


"""
import os

from distutils.core import setup, Extension
dbxml_home = os.environ['BDBXML_HOME']
	               
pef = Extension('_pyextfun', 
		    sources=['extfun.i', 'extfun.cc'],
		    swig_opts=['-c++'],
		    include_dirs=[dbxml_home+os.sep+"include"])

setup (name = 'pyextfun',
       version = '0.1',
       author      = "SWIG Docs",
       description = """Simple swig example from docs""",
       ext_modules = [ pef ],
       py_modules = ["pyextfun"],
       )
