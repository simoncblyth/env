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


Toolset
---------

* http://www.openvdb.org/download/openvdb_toolset_2013.pdf

::

    tools::csgDifference
    tools::csgIntersection
    tools::csgUnion

    tools::LevelSetSphere

    tools::ParticlesToLevelSet

           * Creates a level set from a list of points with position & radius

    tools::MeshToVolume

           * Requires closed (watertight) model for level set

    tools::VolumeToMesh

           * Mesh any scalar field that has a continuous isosurface (quads and tris)
           * Adaptive, using local curvature
            

    Grid::signedFloodFill


reduce compile time
---------------------

* https://groups.google.com/forum/#!topic/openvdb-forum/LLUeaDB1tgw


createLevelSet
---------------

::

    openvdb-;openvdb-find createLevelSet  # indicates: Sphere,Box,Platonic solids : but no Cylinder


* :google:`OpenVDB createLevelSet Cylinder`


Integrations
---------------

Lots of 3D software can import OpenVDB files, 
eg for clouds, fire, effects etc..

Houdini
~~~~~~~~~

Houdini for Astronomy 

* http://www.ytini.com/getstarted.html

Related
----------

yt 
~~~~

* https://bitbucket.org/yt_analysis/yt
* http://yt-project.org/#getyt

yt is an open-source, permissively-licensed python package for analyzing and
visualizing volumetric data.

yt supports structured, variable-resolution meshes, unstructured meshes, and
discrete or sampled data such as particles. Focused on driving
physically-meaningful inquiry, yt has been applied in domains such as
astrophysics, seismology, nuclear engineering, molecular dynamics, and
oceanography. Composed of a friendly community of users and developers, we want
to make it easy to use and develop — we'd love it if you got involved!


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



OpenVDB Intro : the grid contains SDF values in narrow brand around isosurface distance 0
--------------------------------------------------------------------------------------------

::

    using namespace openvdb;
    void makeCylinder(FloatGrid::Ptr grid, float radius, const CoordBBox& indexBB, double h, float backgroundValue)
    {
      typename FloatGrid::Accessor accessor = grid->getAccessor();

      // outputGrid voxel sizes
      for (Int32 i = indexBB.min().x(); i <= indexBB.max().x(); ++i) {
        for (Int32 j = indexBB.min().y(); j <= indexBB.max().y(); ++j) {
          for (Int32 k = indexBB.min().z(); k <= indexBB.max().z(); ++k) {
            Vec3d p(i * h, j * h, k * h);

            float distance = sqrt(p.x() * p.x() + p.y() * p.y() + p.z() * p.z()) - radius;

            if (fabs(distance) < backgroundValue)
              accessor.setValue(Coord(i, j, k), distance);
          }
        }
      }

      grid->signedFloodFill();

      grid->setTransform(openvdb::math::Transform::createLinearTransform(h));
    }

    void createAndWriteGrid()
    {
      float backgroundValue = 2.0;
      openvdb::FloatGrid::Ptr grid = openvdb::FloatGrid::create(backgroundValue);

      CoordBBox indexBB(Coord(-20, -20, -20), Coord(20, 20, 20));
      makeCylinder(grid, 5.0f, indexBB, 0.5, backgroundValue);

      grid->setName("LevelSetSphere");

      openvdb::io::File file("mygrids.vdb");

      openvdb::GridPtrVec grids;
      grids.push_back(grid);

      file.write(grids);
      file.close();
    }



Below from openvdb source, simply set the distance values ...
nothing to stop you parameterizing around the sphere, 
setting distances at the grid points nearby...

Could do the same for CSG composites...


openvdb/tools/LevelSetSphere.h::

    131     typename GridT::Ptr getLevelSet(ValueT voxelSize, ValueT halfWidth)
    132     {
    133         mGrid = createLevelSet<GridT>(voxelSize, halfWidth);
    134         this->rasterSphere(voxelSize, halfWidth);
    135         mGrid->setGridClass(GRID_LEVEL_SET);
    136         return mGrid;
    137     }
    138 
    139 private:
    140     void rasterSphere(ValueT dx, ValueT w)
    141     {
    ...
    145         // Define radius of sphere and narrow-band in voxel units
    146         const ValueT r0 = mRadius/dx, rmax = r0 + w;
    147 
    148         // Radius below the Nyquist frequency
    149         if (r0 < 1.5f)  return;
    150 
    151         // Define center of sphere in voxel units
    152         const Vec3T c(mCenter[0]/dx, mCenter[1]/dx, mCenter[2]/dx);
    153 
    154         // Define index coordinates and their respective bounds
    155         openvdb::Coord ijk;
    156         int &i = ijk[0], &j = ijk[1], &k = ijk[2], m=1;
    157         const int imin=math::Floor(c[0]-rmax), imax=math::Ceil(c[0]+rmax);
    158         const int jmin=math::Floor(c[1]-rmax), jmax=math::Ceil(c[1]+rmax);
    159         const int kmin=math::Floor(c[2]-rmax), kmax=math::Ceil(c[2]+rmax);
    160 
    161         // Allocate a ValueAccessor for accelerated random access
    162         typename GridT::Accessor accessor = mGrid->getAccessor();
    163 
    164         if (mInterrupt) mInterrupt->start("Generating level set of sphere");
    165         // Compute signed distances to sphere using leapfrogging in k
    166         for ( i = imin; i <= imax; ++i ) {
    167             if (util::wasInterrupted(mInterrupt)) return;
    168             const float x2 = math::Pow2(i - c[0]);
    169             for ( j = jmin; j <= jmax; ++j ) {
    170                 const float x2y2 = math::Pow2(j - c[1]) + x2;
    171                 for (k=kmin; k<=kmax; k += m) {
    172                     m = 1;
    173                     /// Distance in voxel units to sphere
    174                     const float v = math::Sqrt(x2y2 + math::Pow2(k-c[2]))-r0,
    175                         d = math::Abs(v);
    176                     if ( d < w ){ // inside narrow band
    177                         accessor.setValue(ijk, dx*v);// distance in world units
    178                     } else {// outside narrow band
    179                         m += math::Floor(d-w);// leapfrog
    180                     }
    181                 }//end leapfrog over k
    182             }//end loop over j
    183         }//end loop over i





macos build using macports
-----------------------------

* http://kirilllykov.github.io/blog/2013/02/04/openvdb-installation-on-macos/

* boost, tbb, openexr, cppunit, glfw


tbb
~~~~

Threading Building Blocks
(Intel TBB)
2017 now under Apache 2.0 license



openexr
---------


::

    simon:openvdb blyth$ port info openexr
    Warning: port definitions are more than two weeks old, consider updating them by running 'port selfupdate'.
    openexr @2.2.0_1 (graphics)
    Sub-ports:            ilmbase, py27-pyilmbase, py34-pyilmbase, py35-pyilmbase, openexr_viewers
    Variants:             universal

    Description:          OpenEXR is a high dynamic-range (HDR) image file format developed by Industrial Light & Magic for use in computer imaging applications.
    Homepage:             http://www.openexr.com

    Build Dependencies:   pkgconfig
    Library Dependencies: ilmbase
    Platforms:            darwin
    License:              BSD
    Maintainers:          Email: mcalhoun@macports.org
                          Policy: openmaintainer


    simon:opticks blyth$ port contents openexr
    Warning: port definitions are more than two weeks old, consider updating them by running 'port selfupdate'.
    Port openexr contains:
      /opt/local/bin/exrenvmap
      /opt/local/bin/exrheader
      /opt/local/bin/exrmakepreview
      /opt/local/bin/exrmaketiled
      /opt/local/bin/exrmultipart
      /opt/local/bin/exrmultiview
      /opt/local/bin/exrstdattr
      /opt/local/include/OpenEXR/ImfAcesFile.h
      /opt/local/include/OpenEXR/ImfArray.h
      /opt/local/include/OpenEXR/ImfAttribute.h
      /opt/local/include/OpenEXR/ImfB44Compressor.h
      /opt/local/include/OpenEXR/ImfBoxAttribute.h
      /opt/local/include/OpenEXR/ImfCRgbaFile.h

    simon:opticks blyth$ port contents ilmbase
    Warning: port definitions are more than two weeks old, consider updating them by running 'port selfupdate'.
    Port ilmbase contains:
      /opt/local/include/OpenEXR/Iex.h
      /opt/local/include/OpenEXR/IexBaseExc.h
      /opt/local/include/OpenEXR/IexErrnoExc.h
      /opt/local/include/OpenEXR/IexExport.h
      /opt/local/include/OpenEXR/IexForward.h
      /opt/local/include/OpenEXR/IexMacros.h




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

openvdb-find(){ openvdb-cd ; find . -type f -exec grep -H ${1:-createLevelSet}  {} \; ; }


openvdb-prefix(){ echo $LOCAL_BASE/openvdb ; }
openvdb-bdir(){   $(openvdb-prefix)/build ; }


openvdb-cmake(){ 

cat << EOX

export GLFW3_ROOT=$HOME/systems/glfw/v3.1.1
export BOOST_ROOT=$HOME/systems/boost/v1.57.0
export TBB_ROOT=$HOME/systems/tbb/tbb44_20151115oss
export ILMBASE_ROOT=/opt/local/include/OpenEXR
export OPENEXR_ROOT=$HOME/systems/OpenEXR/v2.2.0
export BLOSC_ROOT=$HOME/systems/blosc/v1.7.0

cmake -Wno-dev \
    -D OPENEXR_NAMESPACE_VERSIONING=OFF \
    -D CMAKE_CXX_FLAGS="-fPIC -std=c++11" \
    -D TBB_LIBRARY_DIR=$TBB_ROOT/lib \
    -D DOXYGEN_SKIP_DOT=ON \
    -D Blosc_USE_STATIC_LIBS=ON \
    -D USE_GLFW3=ON \
    -D GLFW3_USE_STATIC_LIBS=ON \
    -D Boost_USE_STATIC_LIBS=ON \
    -D CMAKE_INSTALL_PREFIX=$HOME/systems/OpenVDB/v4.0.0 \
    -G "Eclipse CDT4 - Unix Makefiles" \

EOX


}








