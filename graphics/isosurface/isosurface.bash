# === func-gen- : graphics/isosurface/isosurface fgp graphics/isosurface/isosurface.bash fgn isosurface fgh graphics/isosurface
isosurface-src(){      echo graphics/isosurface/isosurface.bash ; }
isosurface-source(){   echo ${BASH_SOURCE:-$(env-home)/$(isosurface-src)} ; }
isosurface-vi(){       vi $(isosurface-source) ; }
isosurface-env(){      elocal- ; }
isosurface-usage(){ cat << EOU

Isosurface Extraction Notes
==============================

Strategy 
----------

Conversion of CSG to BREP only needs to be done 
once for a geometry so it does not need to 
be particularly fast. Hence doing it on CPU 
makes more sense as being more widely applicable.


SDF : Signed Distance Functions and CSG
------------------------------------------

* http://www.iquilezles.org/www/articles/distfunctions/distfunctions.htm

* http://aka-san.halcy.de/distance_fields_prefinal.pdf
* ~/opticks_refs/Procedural_Modelling_with_Signed_Distance_Functions_Thesis.pdf

* p25.. distance functions for sphere, torus, cylinder, cone, box

CSG Combining signed distance functions ?

* Union,  min(dA,dB)
* Intersection, max(dA,dB)
* Difference, max(dA,-dB)   (difference is intersection of A with complement of B)

* Complement negates the distance function


Scene defined by an SDF 

* http://jamie-wong.com/2016/07/15/ray-marching-signed-distance-functions/



Morton Codes for SDF cache ?
-------------------------------

* http://www.forceflow.be/2013/10/07/morton-encodingdecoding-through-bit-interleaving-implementations/
* http://graphics.cs.kuleuven.be/publications/BLD13OCCSVO/
* http://graphics.cs.kuleuven.be/publications/BLD13OCCSVO/BLD13OCCSVO_paper.pdf
* ~/opticks_refs/Sparse_Octree_Morton_BLD13OCCSVO_paper.pdf 


Morton3D
----------

* https://devblogs.nvidia.com/parallelforall/thinking-parallel-part-iii-tree-construction-gpu/

::

    // 1 << 10 = 1024 = 2^10
    //
    // Expands a 10-bit integer into 30 bits
    // by inserting 2 zeros after each bit.
    unsigned int expandBits(unsigned int v)
    {
        v = (v * 0x00010001u) & 0xFF0000FFu;
        v = (v * 0x00000101u) & 0x0F00F00Fu;
        v = (v * 0x00000011u) & 0xC30C30C3u;
        v = (v * 0x00000005u) & 0x49249249u;
        return v;
    }

    // Calculates a 30-bit Morton code for the
    // given 3D point located within the unit cube [0,1].
    unsigned int morton3D(float x, float y, float z)
    {
        x = min(max(x * 1024.0f, 0.0f), 1023.0f);
        y = min(max(y * 1024.0f, 0.0f), 1023.0f);
        z = min(max(z * 1024.0f, 0.0f), 1023.0f);
        unsigned int xx = expandBits((unsigned int)x);
        unsigned int yy = expandBits((unsigned int)y);
        unsigned int zz = expandBits((unsigned int)z);
        return xx * 4 + yy * 2 + zz;
    }


Octree
--------

* https://geidav.wordpress.com/2014/08/18/advanced-octrees-2-node-representations/


Octree GPU Texture 
-------------------

* http://http.developer.nvidia.com/GPUGems2/gpugems2_chapter37.html


Summary List of Techniques
----------------------------

* https://swiftcoder.wordpress.com/planets/isosurface-extraction/


MDC : Manifold Dual Contouring
--------------------------------

* https://github.com/Lin20/isosurface/blob/master/Isosurface/Isosurface/ManifoldDC/Octree.cs


Contouring
-------------

* https://github.com/search?q=contouring


Nick Gildea : Voxels : Dual Contouring Sample
-----------------------------------------------

* http://www.frankpetterson.com/publications/dualcontour/dualcontour.pdf
* http://ngildea.blogspot.tw/2014/11/implementing-dual-contouring.html
* https://github.com/nickgildea/DualContouringSample


Mesh Refinement : surface wavefront propagation
---------------------------------------------------

Semi-Regular Mesh Extraction from Volumes
Zoe J. Wood et al

* http://www.multires.caltech.edu/pubs/meshextraction.pdf


Marching Cubes (MC) vs Extended Marching Cubes (EMC) vs Dual Contouring  (DC)
----------------------------------------------------------------------------------

MC
   rounded edges

EMC
   pyramids along edges, requiring refinement



Nice comparison article 

* http://procworld.blogspot.tw/2010/11/from-voxels-to-polygons.html 

    Marching Cubes is a not good framework to add this since it can only consider
    points that lay along the voxel edges. In this case we clearly need to position
    points anywhere inside the voxel's space. There is another method that does
    exactly this: Dual Contouring.

    Dual Contouring looks at each edge of the voxel at a time. If the corners
    around  the edge have different signs, meaning one is inside the volume and the
    other is outside, the surface is crossing the edge.

    Now, each edge has four neighbor voxels. If the surface is crossing the edge,
    it also means that it necessarily is crossing the four neighbor voxels. You can
    say there is at least one point inside each voxel that lays in the surface. If
    you then output a quadrilateral using the points for the four neighboring
    voxels, you will have a tiny square patch of the surface. Repeating this for
    every edge in the voxel data produces the entire surface.


* http://www.oocities.org/tzukkers/isosurf/isosurfaces.html

    Consider the case of a grid cube containing the corner point of an implicit box
    surface. There are three intersection points of the grid cube with the contour.
    The classic Marching Cubes algorithm connects these points to form the triangle
    approximating the corner. When the normal vector for each point is known, three
    planes can be computed. These three planes together form the corner point, and
    indeed the intersection of three planes is a point.

* http://mathworld.wolfram.com/SingularValueDecomposition.html



Adaptive Marching Cubes
------------------------

* :google:`adaptive marching cubes`

* http://www.acm.org/search?sort=date%3AD%3AS%3Ad1&q=Marching+Cubes&start=0

Left Field Ray Marching With OptiX Approach from SpaceX
----------------------------------------------------------

* http://www.sc15.supercomputing.org/sites/all/themes/SC15images/sci_vis/sci_vis_files/svs111s3-file4.pdf
* -/opticks_refs/Extreme_Multi_Resolution_Viz_SpaceX_svs111s3-file4.pdf


OpenVDB : Dreamworks (MPL)
----------------------------

* see also openvdb- 

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
* -/opticks_refs/OpenVDB_Dreamworks_Museth_TOG13.pdf

* https://github.com/dreamworksanimation/openvdb/blob/b74aa6fc53b3561b8c5f6865617db56c18373a9b/openvdb/unittest/TestVolumeToMesh.cc

Uniformly mesh any scalar grid that has a continuous isosurface.::

    void volumeToMesh   (   const GridType &    grid,
    std::vector< Vec3s > &  points,
    std::vector< Vec4I > &  quads,
    double  isovalue = 0.0 
    )   



Efficient Marching Cubes With Topological Guarantees
-------------------------------------------------------

* https://www-s.ks.uiuc.edu/Research/vmd/projects/ece498/surf/lewiner.pdf
* ~/opticks_refs/Efficient_Marching_Cubes_Topological_Guarantees_lewiner.pdf
* paper has broken link to C++ implementation
* http://www.acm.org/jgt/papers/LewinerEtAl03

* :google:`Thomas Lewiner Marching Cubes` 

An imp is incorporated into scikit-image, which has liberal license:

* https://github.com/scikit-image/scikit-image/pull/2052
* https://github.com/scikit-image/scikit-image/pull/2052/files
* https://github.com/scikit-image/scikit-image/blob/master/LICENSE.txt


That pull is a cython port of the Lewiner C++ imp and includes 
MarchingCubes.cpp as dead code.

* https://github.com/scikit-image/scikit-image/tree/master/skimage/measure/mc_meta/
* https://github.com/scikit-image/scikit-image/blob/master/skimage/measure/mc_meta/MarchingCubes.cpp

Looking for live code:

* :google:`lewiner MarchingCubes.cpp` 


* https://github.com/nci/drishti/blob/master/tools/paint-graphcut/marchingcubes.h

Marching Cubes 33
--------------------


* :google:`marching cubes 33`

* http://liscustodio.github.io/C_MC33/
* http://www.sci.utah.edu/-etiene/pdf/mc33.pdf

    In particular, the work of Etiene et al. [7] shows that the MC33 im-
    plementation by Lewiner et al. [14, 15], fails to produce topo- logically
    correct isosurfaces. 

    As we studied the MC33 implementation, we realized that the source of the
    problem was not merely implementation bugs but the core ideas behind the
    implemented algorithm. In this work, we address issues with Chernyaev’s
    original algorithm, its ex- tension, and its implementation. 



Cubical Marching Squares (from NTU)
--------------------------------------

* https://www.csie.ntu.edu.tw/~cyy/publications/papers/Ho2005CMS.pdf
* ~/opticks_refs/Cubical_Marching_Squares_Ho2005CMS.pdf 
* http://graphics.csie.ntu.edu.tw/CMS/  (GPL and at a glance not easy to integrate)

* https://bitbucket.org/GRassovsky/cubical-marching-squares (BSD)

* https://github.com/mkeeter/kokopelli/blob/master/libfab/asdf/cms.c


Transvoxel
-----------

* http://transvoxel.org


Isosurfaces Over Simplicial Partitions of Multiresolution Grids
-----------------------------------------------------------------

* http://faculty.cs.tamu.edu/schaefer/research/iso_simplicial.pdf
* 2010, Josiah Manson and Scott Schaefer


Dual Marching Cubes
---------------------

Dual Marching Cubes: Primal Contouring of Dual Grids

* https://www.cs.rice.edu/-jwarren/papers/dmc.pdf
* -/opticks_refs/jwarren_dual_marching_cubes_dmc.pdf

Dual Contouring
------------------

* https://github.com/aewallin/dualcontouring

Here are implementations of the Dual Contouring algorithm (SIGGRAPH 2002, see
http://www1.cse.wustl.edu/-taoju/research/dualContour.pdf ). The C++ code was
co-developed by Tao Ju and Scott Schaefer, and Java port is done by Jean-Denis
Boudreault.




* http://faculty.cs.tamu.edu/schaefer/research/dualcontour.pdf
* http://www.frankpetterson.com/publications/dualcontour/dualcontour.pdf

* https://people.eecs.berkeley.edu/~jrs/meshpapers/SchaeferWarren2.pdf
* ~/opticks_refs/Dual_Contouring_Secret_Sauce_SchaeferWarren2.pdf 


Tao Ju (Dual Contouring author)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://sourceforge.net/projects/dualcontouring/
* https://sourceforge.net/projects/dualcontouring/

Not very conveniently, this reads in a persisted octree in .sog or .dcf formats.

* https://sourceforge.net/u/taoju/profile/

* http://www1.cse.wustl.edu/~taoju/
* http://www1.cse.wustl.edu/~taoju/research/dualsimp_tvcg.pdf
* http://www1.cse.wustl.edu/~taoju/research/dualContour.pdf

* http://www.cs.wustl.edu/~taoju/cse554/lectures/lect04_Contouring_I.pdf

* ~/opticks_refs/Manifold_Dual_Contouring_2007_dualsimp_tvcg.pdf


GPU Dual Contouring
---------------------

Analysis and Acceleration of High Quality Isosurface Contouring
LEONARDO AUGUSTO SCHMITZ

* https://www.inf.ufrgs.br/~comba/papers/thesis/diss-leonardo.pdf
* ~/opticks_refs/GPU_Isosurface_Schmitz_diss-leonardo.pdf

GPU-based polygonization and optimization for implicit surfaces Junjie Chen · Xiaogang Jin · Zhigang Deng
* http://graphics.cs.uh.edu/website/Publications/2014-TVC-GPUPolygonization.pdf


Lin20 isosurface
-------------------

A project testing and comparing various algorithms for creating isosurfaces.

* https://github.com/Lin20/isosurface


* https://www.reddit.com/r/dualcontouring/comments/3s3xnp/neilsons_dual_marching_cubes_implementation/
* https://github.com/Lin20/isosurface/blob/master/Isosurface/Isosurface/DMCNeilson/DMCN.cs


Github Marching Cubes
------------------------

* https://github.com/search?q=marching+cubes

* https://github.com/smistad/GPU-Marching-Cubes
* https://www.eriksmistad.no/marching-cubes-implementation-using-opencl-and-opengl/


* https://github.com/pmneila/PyMCubes

* https://github.com/lorensen/MCubes

* https://github.com/uranix/mcubes

  Implementation of Dual Marching Cubes with automatic lookup table generation (using eigen3)


PISTON
--------

* see env-;piston-

* http://datascience.dsscale.org
* http://viz.lanl.gov/projects/PISTON.html
* https://github.com/lanl/PISTON
* http://viz.lanl.gov/projects/piston.pdf
* -/opticks_refs/LANL_MarchingCubes_Isosurface_piston.pdf


PyMCubes : Marching Cubes
-----------------------------

* see env-;pymcubes-

GitHub - pmneila/PyMCubes: Marching cubes (and related tools) for Python

* https://github.com/pmneila/PyMCubes


IsoEx : isosurface extraction built on OpenMesh
------------------------------------------------

* see env-;isoex-

* https://www.graphics.rwth-aachen.de/IsoEx/
* https://www.graphics.rwth-aachen.de/software/
* https://www.graphics.rwth-aachen.de/media/resource_files/IsoEx-1.2.tar.gz

The IsoEx package provides some simple classes and algorithm for isosurface
extraction. Its main purpose is to provide a sample implementation of the
Extended Marching Cubes algorithm:

Kobbelt, Botsch, Schwanecke, Seidel, Feature Sensitive Surface Extraction from
Volume Data, Siggraph 2001.


CUDA Marching Cubes
----------------------

Sorting triangle soup is used as example in CUDA/Thrust presentation 

* http://www.nvidia.com/content/PDF/sc_2010/theater/Bell_SC10.pdf


ICESL: A GPU ACCELERATED CSG MODELER AND SLICER
--------------------------------------------------

* http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.680.4008&rep=rep1&type=pdf
* -/opticks_refs/icesl_2013.pdf

Avoids the mesh, applies CSG operations at pixel level within OpenGL shaders
together with A-buffer.


A-buffer
----------

Basically an A-buffer is a simple list of fragments per pixel

* Cyril Crassin (NVIDIA Research) personal blog http://blog.icare3d.org
* http://blog.icare3d.org/2010/06/fast-and-accurate-single-pass-buffer.html

The idea is very simple: Each fragment is written by the fragment shader at
it's position into a pre-allocated 2D texture array (or a global memory region)
with a fixed maximum number of layers. The layer to write the fragment into is
given by a counter stored per pixel into another 2D texture and incremented
using an atomic increment (or addition) operation ( [image]AtomicIncWrap or
[image]AtomicAdd). After the rendering pass, the A-Buffer contains an unordered
list of fragments per pixel with it's size. To sort these fragments per depth
and compose them on the screen, I simply use a single screen filling quad with
a fragment shader. This shader copy all the pixel fragments in a local array
(probably stored in L1 on Fermi), sort them with a naive bubble sort, and then
combine them front-to-back based on transparency.



Marching Cubes to define isosurface
-------------------------------------

Marching cubes: A high resolution 3D surface construction algorithm

* http://dl.acm.org/citation.cfm?id=37422


:google:`CSG BREP marching cubes`


libigl (MPL) has boolean operations on meshes
-----------------------------------------------

* https://github.com/libigl/libigl
* http://libigl.github.io/libigl/tutorial/tutorial.html#marchingcubes
* http://libigl.github.io/libigl/tutorial/tutorial.html#booleanoperationsonmeshes
* http://libigl.github.io/libigl/tutorial/tutorial.html#csgtree

gts (LGPL)
-----------

GNU Triangulated Surface Library

* http://gts.sourceforge.net


Lin20 isosurface
------------------

A project testing and comparing various algorithms for creating isosurfaces.

* https://github.com/Lin20/isosurface



EOU
}
isosurface-dir(){ echo $(local-base)/env/graphics/isosurface/isosurface ; }
isosurface-cd(){  cd $(isosurface-dir); }
isosurface-mate(){ mate $(isosurface-dir) ; }
isosurface-get(){
   local dir=$(dirname $(isosurface-dir)) &&  mkdir -p $dir && cd $dir


   [ ! -d isosurface ] && git clone https://github.com/Lin20/isosurface

}
