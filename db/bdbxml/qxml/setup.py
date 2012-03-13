#!/usr/bin/env python
"""
setup.py file for SWIG example

    python setup.py build_ext --inplace

The `pyextfun` module fails to import standalone, 
must first `import dbxml`

Possibly being more complete in library specification 
like the `dbxml` setup.py can avoid this.
See:
   /usr/local/env/db/dbxml-2.5.16/dbxml/src/python/setup.py

"""
import os

from distutils.core import setup, Extension
dbxml_home = os.environ['BDBXML_HOME']
	               
pef = Extension('_pyextfun', 
		    sources=['extfun.i', 'extfun.cc'],
		    swig_opts=['-Wall','-c++','-threads'],
		    include_dirs=[os.path.join(dbxml_home,"include")])

setup (name = 'pyextfun',
       version = '0.1',
       author      = "SWIG Docs",
       description = """Simple swig example from docs""",
       ext_modules = [ pef ],
       py_modules = ["pyextfun"],
       )
