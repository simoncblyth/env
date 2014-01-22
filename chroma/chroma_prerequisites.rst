Chroma Prerequisites
=====================

* http://chroma.bitbucket.org/install/details.html

Difficult hardware and software requirements, lots of pre-requisites.

* Are they all really, really needed ?
* what are they for ?


Software
---------

From http://chroma.bitbucket.org/install/details.html

#. Python 2.6 or later
#. CUDA 4.1 Toolkit and NVIDIA driver. 
#. Boost::Python
#. Numpy 1.6 or later
#. Pygame
#. Matplotlib
#. uncertainties
#. PyCUDA 2011.2 or later
#. PyUblas
#. ZeroMQ
#. GEANT4.9.5 or later

   * non-cmake 4.9.4 mentioned in setup.py
   * DYB currently at geant4.9.2.p01 see :doc:`/geant4/release_notes`

#. Patched version of g4py (DID I INSTALL THE PATCHED ONE ? STILL NEEDED ?) 
#. ROOT 5.32 or later

   * DYB currently at root_v5.26.00e :doc:`/root/release_notes`

uncertainties
-------------

Derivative based error propagation.

* http://pythonhosted.org/uncertainties/index.html
* http://pythonhosted.org/uncertainties/user_guide.html

::

    simon:chroma blyth$ find . -name '*.*' -exec grep -l uncertainties {} \;
    ./chroma/benchmark.py
    ./chroma/histogram/histogram.py
    ./chroma/histogram/histogramdd.py
    ./chroma/likelihood.py
    ./chroma/parabola.py
    ./setup.py
    ./test/test_parabola.py


pycuda
-------

* http://documen.tician.de/pycuda/


pyublas
---------

* http://documen.tician.de/pyublas/

PyUblas provides a seamless glue layer between Numpy and Boost.Ublas for use with Boost.Python

* http://www.boost.org/doc/libs/1_54_0/libs/numeric/ublas/doc/index.htm
* http://www.netlib.org/blas/

BLAS : Basic Linear Algebra Subprograms


::

    simon:chroma blyth$ find . -name '*.*' -exec grep -l pyublas {} \;
    ./chroma/generator/g4gen.py    # imported but unused here ?
    ./doc/source/chroma-setup.py
    ./doc/source/chroma-setup.sh
    ./setup.py
    ./src/G4chroma.cc


src/G4chroma.cc
~~~~~~~~~~~~~~~~

::

    118 #include <boost/python.hpp>
    119 #include <pyublas/numpy.hpp>
    120 
    121 using namespace boost::python;
    122 
    123 pyublas::numpy_vector<double> PTA_GetX(const PhotonTrackingAction *pta)
    124 {
    125   pyublas::numpy_vector<double> r(pta->GetNumPhotons());
    126   pta->GetX(&r[0]);
    127   return r;
    128 }



pygame
--------

Python layer ontop of SDL, used for GUI and "game" control.

Pygame is implemented as a mixture of Python, C and Assembler code, wrapping
3rd party libraries with CPython API interfaces. 

* http://www.pygame.org/news.html
* https://bitbucket.org/pygame/pygame
* http://en.wikipedia.org/wiki/Pygame

::

    simon:chroma blyth$ pwd
    /usr/local/env/muon_simulation/chroma/muon_simulation/chroma
    simon:chroma blyth$ find . -name '*.py' -exec grep -l pygame {} \;
    ./chroma/__init__.py    # does not insist on pygame available
    ./chroma/camera.py      # used for GUI, keyboard event handling 
    ./doc/source/chroma-setup.py    # brittle yum/apt/pip/cmake based installer for most everything 
    ./setup.py


spnav
-----

The spnav module provides a Python interface to the libspnav C library, which
allows you to read events from a Space Navigator 3D mouse on Linux systems.
These input devices simultaneously report linear force and rotational torque
applied by the user to the device, along with button events. See:


* http://spnav.readthedocs.org/en/latest/
* https://pypi.python.org/pypi/spnav
* http://www.3dconnexion.com/products/spacenavigator.html


pyzmq 
------

Underlying ZeroMQ is now auto-built by the python setup.py installation, no longer need pyzmq-static.

* https://pypi.python.org/pypi/pyzmq/
* https://pypi.python.org/pypi/pyzmq-static/



 
