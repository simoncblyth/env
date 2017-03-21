# === func-gen- : graphics/openvdb/openvdb fgp graphics/openvdb/openvdb.bash fgn openvdb fgh graphics/openvdb
openvdb-src(){      echo graphics/openvdb/openvdb.bash ; }
openvdb-source(){   echo ${BASH_SOURCE:-$(env-home)/$(openvdb-src)} ; }
openvdb-vi(){       vi $(openvdb-source) ; }
openvdb-env(){      elocal- ; }
openvdb-usage(){ cat << EOU

OpenVDB : Dreamworks (MPL)
============================

* Open sourced by Dreamworks in 2012

* http://www.openvdb.org
* http://www.openvdb.org/documentation/
* https://github.com/dreamworksanimation/openvdb
* http://www.openvdb.org/documentation/doxygen/overview.html

* https://github.com/dreamworksanimation/openvdb/search?q=CSG

* http://ken.museth.org/OpenVDB.html

OpenVDB is an Academy Award winning open sourced C++ library comprising a
hierarchical data structure and a suite of tools for the efficient manipulation
of sparse, possibly time-varying, volumetric data discretized on a
three-dimensional grid. It is based on VDB (aka DB+Grid), which was developed
by Ken Museth at DreamWorks Animation, and it offers an effectively infinite 3D
index space, compact storage (both in memory and on disk), fast data access
(both random and sequential), and a collection of algorithms specifically
optimized for the data structure for common tasks such as filtering, CSG,
compositing, numerical simulations, sampling and voxelization from other
geometric representations. The technical details of VDB are described in the
paper “VDB: High-Resolution Sparse Volumes with Dynamic Topology”. See press
releases by DreamWorks, Digital Domain and SideFX or visit the openvdb site.

* http://ken.museth.org/OpenVDB_files/Museth_TOG13.pdf
* ~/opticks_refs/OpenVDB_Dreamworks_Museth_TOG13.pdf


Uniformly mesh any scalar grid that has a continuous isosurface.::

    void volumeToMesh   (   const GridType &    grid,
    std::vector< Vec3s > &  points,
    std::vector< Vec4I > &  quads,
    double  isovalue = 0.0 
    )   


Examples
----------

* http://www.openvdb.org/documentation/doxygen/codeExamples.html


* :google:`OpenVDB Tutorial`

* https://callumjamesjames.wordpress.com/2014/08/03/openvdb-tutorials-now-available/


Level Sets
------------

* My take, level sets are essentially a small band range around an isovalue


Nice intro to level sets and OpenVDB

* http://kirilllykov.github.io/blog/2013/04/02/level-set-openvdb-intro-1/

::

    Note, that the call of the method signedFloodFill() propagates the sign from
    initialized grid points to the uninitialized, since the background value is set
    in the grid by module. signedFloodFill() might be used on closed surfaces only,
    so I picked up a sphere instead of a cylinder.


macos build using macports
-----------------------------

* http://kirilllykov.github.io/blog/2013/02/04/openvdb-installation-on-macos/

* boost, tbb, openexr, cppunit, glfw


pyopenvdb
-----------

* http://www.openvdb.org/documentation/doxygen/python.html

The OpenVDB Python module can optionally be compiled with NumPy support. With
NumPy enabled, the copyFromArray and copyToArray grid methods can be used to
exchange data efficiently between scalar-valued grids and three-dimensional
NumPy arrays and between vector-valued grids and four-dimensional NumPy arrays.


When copying from a NumPy array, values in the array that are equal to the
destination grid’s background value (or close to it, if the tolerance argument
to copyFromArray is nonzero) are set to the background value and are marked
inactive. All other values are marked active.



Externals
----------

OpenEXR
~~~~~~~~~~~

* http://www.openexr.com  
* https://github.com/openexr/openexr

OpenEXR is a high dynamic-range (HDR) image file format developed by Industrial
Light & Magic for use in computer imaging applications.

OpenEXR is used by ILM on all motion pictures currently in production. The
first movies to employ OpenEXR were Harry Potter and the Sorcerers Stone, Men
in Black II, Gangs of New York, and Signs. Since then, OpenEXR has become ILM's
main image file format.


Blosc : fast decompression
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* http://www.blosc.org
* https://github.com/Blosc
* https://github.com/Blosc/c-blosc
* https://github.com/Blosc/c-blosc2




EOU
}
openvdb-dir(){ echo $(local-base)/env/graphics/openvdb/openvdb ; }
openvdb-cd(){  cd $(openvdb-dir); }
openvdb-get(){
   local dir=$(dirname $(openvdb-dir)) &&  mkdir -p $dir && cd $dir

   [ ! -d "openvdb" ] && git clone https://github.com/dreamworksanimation/openvdb
}
