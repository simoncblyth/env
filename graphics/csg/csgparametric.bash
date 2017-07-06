# === func-gen- : graphics/csg/csgparametric fgp graphics/csg/csgparametric.bash fgn csgparametric fgh graphics/csg
csgparametric-src(){      echo graphics/csg/csgparametric.bash ; }
csgparametric-source(){   echo ${BASH_SOURCE:-$(env-home)/$(csgparametric-src)} ; }
csgparametric-vi(){       vi $(csgparametric-source) ; }
csgparametric-env(){      elocal- ; }
csgparametric-usage(){ cat << EOU

CSG Parametric Approaches
===========================

* :google:`CSG Parametric Solid Algorithm`

Overview
----------

* perhaps could just apply the technique to IM meshes for the primitives, 
  prior to implementing parametric meshes for them : this seems workable
  as use the implicits to find the intersecting faces

  But the mesh needs to be amenable to subdivision, to handle multi-cross edged, 
  so probably easier just to use exact parametric meshes.

* current meshing just uses the CSG tree for the SDF evaluation producing 
  a single mesh, this approach needs intermediate meshes at each 
  node of the CSG tree 


CGAL (GPL)
--------------

3D Boolean Operations on Nef Polyhedra

* http://doc.cgal.org/latest/Nef_3/index.html


quickcsg
---------

* http://kinovis.inrialpes.fr
* http://kinovis.inrialpes.fr/quickcsg/
* https://hal.inria.fr/hal-01121419
* https://hal.inria.fr/hal-01121419/document
* http://kinovis.inrialpes.fr/static/QuickCSG/
* ~/opticks_refs/QuickCSG_Kinovis_RR-8687.pdf




OpenSCAD : edge coincidence between sub-objects
--------------------------------------------------

* https://github.com/openscad/openscad/issues/131


CNRG
-----

* http://www.cc.gatech.edu/~jarek/papers/CNRG.pdf
* ~/opticks_refs/CSG_Non_Regularized_Jarek_Requicha_CNRG.pdf


csg.js
--------

* http://evanw.github.io/csg.js/docs/


Subtraction and intersection naturally follow from set operations.
with ~ the complement operator.

UNION
     A | B

SUBTRACTION
     A - B 
     ~(~A | B) 

INTERSECTION  
     A & B 
     ~(~A | ~B)



Cork
-----

* https://github.com/gilbo/cork


CSG BBox ?
-------------

* :google:`CSG intersection bbox`




Philip Rideout : pbrt plugin to perform CSG intersection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* http://prideout.net/archive/pbrt/csg.cpp


CSG BSP
----------

* https://github.com/mkkellogg/CSG-BSP
* https://github.com/dabroz/csgjs-cpp

Moka
-----

* http://liris.cnrs.fr/moka/operations-architecture.php
* http://moka-modeller.sourceforge.net/docs/librairies/html/boolean-operations_8hh.html


MEPP
-----

.. mostly based on the Polyhedron type from CGAL ..

* ~/opticks_refs/MEPP_Grapp_2012.pdf
* https://liris.cnrs.fr/mepp/pdfs/MEPP_Grapp_2012.pdf


* ~/opticks_refs/Exact_Efficient_Booleans_for_Polyhedra_Liris-4883.pdf
* http://liris.cnrs.fr/Documents/Liris-4883.pdf





Hybrid Booleans
------------------

* https://www.graphics.rwth-aachen.de/media/papers/boolean_021.pdf
* ~/opticks_refs/Hybrid_Booleans_RWTH_boolean_021.pdf


Mesh Matching
--------------

* http://imr.sandia.gov/papers/imr17/Staten.pdf


Advancing Front
-----------------

* http://www.iue.tuwien.ac.at/phd/fleischmann/node39.html




Afront (GPL)
-------------------------

* https://sourceforge.net/projects/afront/
* http://afront.sourceforge.net

Afront is a tool for meshing and remeshing surfaces. The main application of
Afront is the generation of high-quality meshes from a variety of surface
descriptions, from triangle meshes themselves (remeshing) to implicit surfaces
to point set surfaces.

Ear Clipping Triangulation
---------------------------

* :google:`mesh fill triangle fan`
* https://computergraphics.stackexchange.com/questions/4741/turn-an-enclosed-region-into-a-triangle-mesh/4747
* https://www.geometrictools.com/Documentation/TriangulationByEarClipping.pdf
* ~/opticks_refs/TriangulationByEarClipping.pdf

* https://www.geometrictools.com/GTEngine/Include/Mathematics/GteTriangulateEC.h
* https://www.geometrictools.com/GTEngine/Include/Mathematics/GteTriangulateCDT.h

* https://www.geometrictools.com/Source/ComputationalGeometry.html


Marching Triangles
--------------------

* https://en.wikipedia.org/wiki/Delaunay_triangulation
* http://fab.cba.mit.edu/classes/S62.12/docs/Hilton_marching_triangles.pdf
* ~/opticks_refs/Hilton_marching_triangles.pdf

* https://scicomp.stackexchange.com/questions/25875/is-the-marching-triangles-algorithm-guaranteed-to-terminate

* http://www.graphicon.ru/html/2009/conference/se1/14/14_Paper.pdf
* ~/opticks_refs/improved_marching_triangle_14_Paper.pdf


Adaptive Mesh Booleans, Autodesk Research
--------------------------------------------

* https://arxiv.org/pdf/1605.01760.pdf
* ~/opticks_refs/Adaptive_Mesh_Booleans_Autodesk_Research_1605.01760.pdf


Boolean Operations with Implicit and Parametric Representation of Primitives Using R-Functions
------------------------------------------------------------------------------------------------

Yohan D. Fougerolle, Andrei Gribok, Sebti Foufou,
Frederic Truchetet, Member, IEEE, and Mongi A. Abidi, Member, IEEE

* https://ai2-s2-pdfs.s3.amazonaws.com/bbed/e04dc9213207cb713f7d0a312c41c33072b9.pdf
* ~/opticks_refs/Boolean_Operations_With_Implicit_and_Parametric_reps_using_R_functions.pdf
* http://ieeexplore.ieee.org/document/1471690/

* http://dl.acm.org/citation.cfm?id=1080028

We present a new and efficient algorithm to accurately polygonize an implicit
surface generated by multiple Boolean operations with globally deformed
primitives. Our algorithm is special in the sense that it can be applied to
objects with both an implicit and a parametric representation, such as
superquadrics, supershapes, and Dupin cyclides. The input is a Constructive
Solid Geometry tree (CSG tree) that contains the Boolean operations, the
parameters of the primitives, and the global deformations. At each node of the
CSG tree, the implicit formulations of the subtrees are used to quickly
determine the parts to be transmitted to the parent node, while the primitives’
parametric definition are used to refine an intermediary mesh around the
intersection curves. The output is both an implicit equation and a mesh
representing its solution. For the resulting object, an implicit equation with
guaranteed differential properties is obtained by simple combinations of the
primitives’ implicit equations using R-functions. Depending on the chosen
R-function, this equation is continuous and can be differentiable everywhere.
The primitives’ parametric representations are used to directly polygonize the
resulting surface by generating vertices that belong exactly to the zero-set of
the resulting implicit equation. The proposed approach has many potential
applications, ranging from mechanical engineering to shape recognition and data
compression. Examples of complex objects are presented and commented on to show
the potential of our approach for shape modeling.


R-functions Rvachev
~~~~~~~~~~~~~~~~~~~~~~

* https://en.m.wikipedia.org/wiki/Rvachev_function


* http://spatial.engr.wisc.edu/?page_id=435

* ~/opticks_refs/Shapiro-Rvachev-Functions.pdf
* http://spatial.engr.wisc.edu/wp-uploads/2014/04/0000-1.pdf

  Shapiro primer 


Implicit Surface Modeling using Supershapes and R-functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Shorter paper on same topic.

* http://le2i.cnrs.fr/IMG/publications/PG05.pdf
* ~/opticks_refs/Implicit_Parametric_Fougerolle_PG05.pdf



Algorithm 1 Recursive node evaluation

Input: two CSG subtrees
Output: Closed mesh transferred to the parent node

1. Inside/outside evaluation using the subtrees' implicit functions
2. Create intersection curves

::

     for all Intersecting faces do
         Create intersection points I such as F(I) < epsilon
         Build intersection curves 
     end for

3. Re-sample intersections curve to assert d(In,In+1) < delta, 
   where In and In+1 are consecutive intersection points
4. Split faces
5. Merge intersection curves
6. Transfer vertices and faces to parent node 






BOOLE
-------

BOOLE : A Boundary Evaluation System for Boolean combinations of Sculptured Solids
S Krishnan

* http://people.mpi-inf.mpg.de/~schoemer/ECG/SS02/papers/boole2.pdf

In this paper we describe a system BOOLE that generates the boundary
representations reps of solids given as a CSG expression in the form of
trimmed Bezier patches.



CSG Parametric Solid Algorithm
---------------------------------

* http://web.iitd.ac.in/~hegde/cad/lecture/L32_solidmcsg.pdf

Summary of a CSG algorithm

* D & C (Divide and Conquer)
* neighborhood classification: Face/Edge/Vertex

The following steps describe a general CSG algorithm 
based on divide and conquer approach:

1. Generate a sufficient number of t-faces, set of faces of participating primitives, say A and B.
2. Classify self edges of A w.r.t A including neighborhood.
3. Classify self edges of A w.r.t B using D & C paradigm. 
   If A or B is not primitive then this step is followed recursively.
4. Combine the classifications in step 2 and 3 via Boolean operations.
5. Regularize the ‘on’ segment that result from step 4 
   discarding the segments that belong to only one face of S.
6. Store the final ‘on’ segments that result from step 5 as 
   part of the boundary of S. 
   Steps 2 to 6 is performed for each of t-edge of a given t-face of A.
7. Utilize the surface/surface intersection to find cross edges that 
   result from intersecting faces of B (one at a time) 
   with the same t-face mentioned in step 6.
8. Classify each cross edge w.r.t S by repeating 
   steps 2 to 4 with the next self edge of A.
9. Repeat steps 5 and 6 for each cross edge
10. Repeat steps 2 to 9 for each t-face of A.
11. Repeat stpes 2 to 6 for each t-face of B.







EOU
}
csgparametric-dir(){ echo $(local-base)/env/graphics/csg/graphics/csg-csgparametric ; }
csgparametric-cd(){  cd $(csgparametric-dir); }
csgparametric-mate(){ mate $(csgparametric-dir) ; }
csgparametric-get(){
   local dir=$(dirname $(csgparametric-dir)) &&  mkdir -p $dir && cd $dir

}
