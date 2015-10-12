# === func-gen- : graphics/csg/csg fgp graphics/csg/csg.bash fgn csg fgh graphics/csg
csg-src(){      echo graphics/csg/csg.bash ; }
csg-source(){   echo ${BASH_SOURCE:-$(env-home)/$(csg-src)} ; }
csg-vi(){       vi $(csg-source) ; }
csg-env(){      elocal- ; }
csg-usage(){ cat << EOU

CSG : Constructive Solid Geometry
==================================


* http://www.doc.ic.ac.uk/~dfg/graphics/graphics2008/GraphicsLecture10.pdf

* http://web.cse.ohio-state.edu/~parent/classes/681/Lectures/19.RayTracingCSG.pdf

* https://mit-crpg.github.io/OpenMOC/

* https://devtalk.nvidia.com/default/topic/771034/optix/constructive-solid-geometry/

* http://www.cs.utah.edu/~shirley/books/fcg2/rt.pdf

* https://www.clear.rice.edu/comp360/lectures/SurfSpeText.pdf

  Intersections of lines and Special Surfaces: cone, sphere, ...

* https://www.cg.tuwien.ac.at/courses/Rendering/VU.SS2015.html

  CG course with YouTube videos  


XRT Renderer : simpler? way of handling CSG trees
---------------------------------------------------

* http://xrt.wdfiles.com/local--files/doc%3Acsg/CSG.pdf

* http://xrt.wikidot.com/search:site/q/csg 

* http://xrt.wikidot.com/doc:csg



CSG Simplification
-----------------------------------------------

* http://webserver2.tecgraf.puc-rio.br/~lhf/ftp/doc/sib2006b.pdf

  Spatial Partitioning to simplify CSG rendering
  Hardware-assisted Rendering of CSG Models


* http://www.cc.gatech.edu/~turk/my_papers/pxpl_csg.pdf

  CSG Tree Normalization and Pruning





CSG Ray Tracing Techniques
-----------------------------

* http://www.sciencedirect.com/science/article/pii/S0734189X86800548

  A new algorithm for object oriented ray tracing  (1986) Saul Youssef 


Ray Tracing CSG Models : implementation details
------------------------------------------------------ 

* http://web.cse.ohio-state.edu/~parent/classes/681/Lectures/19.RayTracingCSG.pdf





Cool WebGL interface allowing to edit CSG geometries
----------------------------------------------------

* http://evanw.github.io/csg.js/
* http://evanw.github.io/csg.js/docs/
* https://github.com/evanw/csg.js/


* http://learningthreejs.com/blog/2011/12/10/constructive-solid-geometry-with-csg-js/

  three.js bridge to csg.js





EOU
}
csg-dir(){ echo $(local-base)/env/graphics/csg/graphics/csg-csg ; }
csg-cd(){  cd $(csg-dir); }
csg-mate(){ mate $(csg-dir) ; }
csg-get(){
   local dir=$(dirname $(csg-dir)) &&  mkdir -p $dir && cd $dir

}
