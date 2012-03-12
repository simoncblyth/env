#!/usr/bin/env python
"""

http://docs.cython.org/src/userguide/wrapping_CPlusPlus.html

Build::

   python setup.py build

Test::
   
    PYTHONPATH=build/lib.macosx-10.5-ppc-2.5 python -c "from rectangle import PyRectangle ; r = PyRectangle(1,2,3,400) ; print r.getArea() "  

Note:

#. use different stem for implementation eg ``RectangleImp.cpp`` and header/pyx otherwise cython stomps on the implementation by generating ``rectangle.cpp`` 

"""
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

setup(ext_modules=[Extension(
                   "rectangle",                 # name of extension
                   ["rectangle.pyx", "RectangleImp.cpp"], #  our Cython source
                   language="c++")],  # causes Cython to create C++ source
      cmdclass={'build_ext': build_ext})
