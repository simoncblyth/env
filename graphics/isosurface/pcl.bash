# === func-gen- : graphics/isosurface/pcl fgp graphics/isosurface/pcl.bash fgn pcl fgh graphics/isosurface
pcl-src(){      echo graphics/isosurface/pcl.bash ; }
pcl-source(){   echo ${BASH_SOURCE:-$(env-home)/$(pcl-src)} ; }
pcl-vi(){       vi $(pcl-source) ; }
pcl-env(){      elocal- ; }
pcl-usage(){ cat << EOU


Point Cloud Library (BSD)
===========================

* http://pointclouds.org
* https://github.com/PointCloudLibrary



The Point Cloud Library (PCL) is a standalone, large scale, open project for 2D/3D image and point cloud processing.

PCL is released under the terms of the BSD license and is open source software. It is free for commercial and research use.

PCL is cross-platform, and has been successfully compiled and deployed on
Linux, MacOS, Windows, and Android. To simplify development, PCL is split into
a series of smaller code libraries, that can be compiled separately. This
modularity is important for distributing PCL on platforms with reduced
computational or size constraints.


Via: http://stackoverflow.com/questions/838761/robust-algorithm-for-surface-reconstruction-from-3d-point-cloud

The library PCL has a module dedicated to surface reconstruction and is in
active development (and is part of Google's Summer of Code). The surface module
contains a number of different algorithms for reconstruction. PCL also has the
ability to estimate surface normals, incase you do not have them provided with
your point data, this functionality can be found in the features module. PCL is
released under the terms of the BSD license and is open source software, it is
free for commercial and research use.


Build
-------

* http://www.pointclouds.org/documentation/tutorials/compiling_pcl_macosx.php


Externals
-----------

Flann (BSD)
~~~~~~~~~~~~

* http://www.cs.ubc.ca/research/flann/

FLANN is a library for performing fast approximate nearest neighbor searches in
high dimensional spaces. It contains a collection of algorithms we found to
work best for nearest neighbor search and a system for automatically choosing
the best algorithm and optimum parameters depending on the dataset.  FLANN is
written in C++ and contains bindings for the following languages: C, MATLAB and
Python.

The FLANN license was changed from LGPL to BSD.


Eigen (MPL2)
~~~~~~~~~~~~~~

Eigen is a C++ template library for linear algebra: matrices, vectors, numerical solvers, and related algorithms.

* http://eigen.tuxfamily.org/index.php?title=Main_Page#License

Eigen is Free Software. Starting from the 3.1.1 version, it is licensed under
the MPL2, which is a simple weak copyleft license. Common questions about the
MPL2 are answered in the official MPL2 FAQ.

Earlier versions were licensed under the LGPL3+.

Note that currently, a few features rely on third-party code licensed under the
LGPL: SimplicialCholesky, AMD ordering, and constrained_cg. Such features can
be explicitly disabled by compiling with the EIGEN_MPL2_ONLY preprocessor
symbol defined. Furthermore, Eigen provides interface classes for various
third-party libraries (usually recognizable by the <Eigen/*Support> header
name). Of course you have to mind the license of the so-included library when
using them.

Virtually any software may use Eigen. For example, closed-source software may
use Eigen without having to disclose its own source code. Many proprietary and
closed-source software projects are using Eigen right now, as well as many
BSD-licensed projects.

See the MPL2 FAQ for more information, and do not hesitate to contact us if you
have any questions.



Surface
---------

http://docs.pointclouds.org/trunk/group__surface.html




Octree
---------


http://docs.pointclouds.org/trunk/group__octree.html







EOU
}
pcl-dir(){ echo $(local-base)/env/graphics/isosurface/graphics/PointCloudLibrary/pcl ; }
pcl-cd(){  cd $(pcl-dir); }
pcl-mate(){ mate $(pcl-dir) ; }
pcl-get(){
   local dir=$(dirname $(pcl-dir)) &&  mkdir -p $dir && cd $dir

   [ ! -d pcl ] && git clone https://github.com/PointCloudLibrary/pcl
}
