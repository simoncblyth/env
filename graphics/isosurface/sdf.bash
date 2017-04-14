# === func-gen- : graphics/isosurface/sdf fgp graphics/isosurface/sdf.bash fgn sdf fgh graphics/isosurface
sdf-src(){      echo graphics/isosurface/sdf.bash ; }
sdf-source(){   echo ${BASH_SOURCE:-$(env-home)/$(sdf-src)} ; }
sdf-vi(){       vi $(sdf-source) ; }
sdf-env(){      elocal- ; }
sdf-usage(){ cat << EOU


SDF : Signed Distance Functions
==================================


Intro
------

* http://www.alanzucconi.com/2016/07/01/signed-distance-functions/
* http://www.iquilezles.org/www/articles/distfunctions/distfunctions.htm


Lots of shapes and modifiers
------------------------------

* http://mercury.sexy/hg_sdf/

  Implementations of a lot of shapes, including platonics 
  and generalized distance functions based on 

  * https://www.viz.tamu.edu/faculty/ergun/research/implicitmodeling/papers/sm99.pdf


Procedural modeling with signed distance functions (Thesis)
-------------------------------------------------------------

* http://aka-san.halcy.de/distance_fields_prefinal.pdf
* ~/opticks_refs/Procedural_Modelling_with_Signed_Distance_Functions_Thesis.pdf

* p25.. distance functions for sphere, torus, cylinder, cone, box



SDF and CSG
------------------------------------------

CSG Combining signed distance functions ?

* Union,  min(dA,dB)
* Intersection, max(dA,dB)
* Difference, max(dA,-dB)   (difference is intersection of A with complement of B)

* Complement negates the distance function


Scene defined by an SDF 

* http://jamie-wong.com/2016/07/15/ray-marching-signed-distance-functions/


TSDF
-----

Code for integrating, raytracing, and meshing a TSDF on the CPU

* https://github.com/sdmiller/cpu_tsdf


Morphing of SDFs
-------------------

Interactive Modeling of Implicit Surfaces using a Direct Visualization Approach with Signed Distance Functions
Tim Reiner, Gregor Muckl, Carsten Dachsbacher 

* https://cg.ivd.kit.edu/downloads/IntModelingSDF.pdf





EOU
}
sdf-dir(){ echo $(local-base)/env/graphics/isosurface/graphics/isosurface-sdf ; }
sdf-cd(){  cd $(sdf-dir); }
sdf-mate(){ mate $(sdf-dir) ; }
sdf-get(){
   local dir=$(dirname $(sdf-dir)) &&  mkdir -p $dir && cd $dir

}
