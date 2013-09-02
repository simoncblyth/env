# === func-gen- : graphics/mesh/meshpy fgp graphics/mesh/meshpy.bash fgn meshpy fgh graphics/mesh
meshpy-src(){      echo graphics/mesh/meshpy.bash ; }
meshpy-source(){   echo ${BASH_SOURCE:-$(env-home)/$(meshpy-src)} ; }
meshpy-vi(){       vi $(meshpy-source) ; }
meshpy-env(){      elocal- ; }
meshpy-usage(){ cat << EOU

MESHPY
======

* http://mathema.tician.de/software/meshpy
* http://documen.tician.de/meshpy/

MeshPy provides Python interfaces to two well-regarded mesh generators, 

#. Triangle by J. Shewchuk http://www.cs.cmu.edu/~quake/triangle.html
#. TetGen by Hang Si. 

Both are included in the package in slightly modified versions.


Background
-----------

* :google:`boundary conforming Delaunay mesh`
* :google:`conforming Delaunay`
* http://www.cis.upenn.edu/grad/documents/siquera.pdf 


TetGen
-------

http://wias-berlin.de/software/tetgen/

If the input boundary contains no acute angle (in practice, this condition can
be relaxed to no input angle smaller than 60 degree), TetGen will generate a
boundary conforming Delaunay mesh. 

Currently, TetGen can directly read and write data in the following file formats:

* .off - Geomview's polyhedral file format.
* .ply - Plyhedral file format.
* .stl - Stereolithography format.
* .mesh - Medit's mesh file format.


Install
-------

The configure.py creates a Makefile::

    simon:meshpy blyth$ ./configure.py --boost-inc-dir=/opt/local/include --boost-lib-dir=/opt/local/lib --python-exe=/opt/local/bin/python --boost-python-libname=boost_python --no-use-shipped-boost
    simon:meshpy blyth$ ./configure.py --boost-inc-dir=/opt/local/include --boost-lib-dir=/opt/local/lib --python-exe=/opt/local/bin/python2.6 --boost-python-libname=boost_python --no-use-shipped-boost

Then standard approach::

    make 
    sudo make install


Must use same python as boost_python 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    simon:examples blyth$ python -c "import meshpy"
    simon:examples blyth$ python -c "import meshpy.tet"
    Fatal Python error: Interpreter not initialized (version mismatch?)
    Abort trap

Saying yes to the report dialog see the source::

    Thread 0 Crashed:
    0   libSystem.B.dylib               0x957659f0 __kill + 12
    1   libSystem.B.dylib               0x95800bf8 abort + 84
    2   org.python.python               0x010f5a50 Py_InitModule4 + 64
    3   libboost_python.dylib           0x0038e970 boost::python::detail::init_module(char const*, void (*)()) + 48
    4   org.python.python               0x00281b80 _PyImport_LoadDynamicModule + 192
    5   org.python.python               0x0027f9c4 import_submodule + 420
    6   org.python.python               0x0027fc6c load_next + 332
    7   org.python.python               0x002803d4 import_module_level + 708

Probably the boost_python is built against a different python::

    simon:chroma blyth$ otool -L /opt/local/lib/libboost_python.dylib 
    /opt/local/lib/libboost_python.dylib:
            /opt/local/lib/libboost_python.dylib (compatibility version 0.0.0, current version 0.0.0)
            /opt/local/Library/Frameworks/Python.framework/Versions/2.6/Python (compatibility version 2.6.0, current version 2.6.0)
            /usr/lib/libstdc++.6.dylib (compatibility version 7.0.0, current version 7.4.0)
            /usr/lib/libgcc_s.1.dylib (compatibility version 1.0.0, current version 1.0.0)
            /usr/lib/libSystem.B.dylib (compatibility version 1.0.0, current version 111.1.7)

Examples
---------

::

    simon:examples blyth$ python2.6 demo.py 
    Mesh Points:
    0 [0.0, 0.0, 0.0]
    1 [2.0, 0.0, 0.0]
    ...
    118 [12, 28, 49, 54]
    119 [49, 28, 29, 54]
    120 [29, 28, 12, 54]
    Traceback (most recent call last):
      File "demo.py", line 23, in <module>
        mesh.write_vtk("test.vtk")
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.6/lib/python2.6/site-packages/MeshPy-2013.1.2-py2.6-macosx-10.5-ppc.egg/meshpy/tet.py", line 104, in write_vtk
        import pyvtk
    ImportError: No module named pyvtk
    simon:examples blyth$ 



pyvtk
-------

* http://code.google.com/p/pyvtk/
* https://pypi.python.org/pypi/PyVTK
* http://www.vtk.org/



EOU
}
meshpy-dir(){ echo $(local-base)/env/graphics/mesh/graphics/meshpy ; }
meshpy-cd(){  cd $(meshpy-dir); }
meshpy-mate(){ mate $(meshpy-dir) ; }
meshpy-get(){
   local dir=$(dirname $(meshpy-dir)) &&  mkdir -p $dir && cd $dir

   git clone http://git.tiker.net/trees/meshpy.git

}
