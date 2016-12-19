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



:google:`GPU CSG boolean Roth`
---------------------------------

Spatially Efficient Tree Layout for GPU Ray-tracing of Constructive Solid Geometry Scenes

* http://ceur-ws.org/Vol-1576/090.pdf


Andrew Kensler
----------------

Ray Tracing CSG Objects Using Single Hit Intersections

* http://xrt.wdfiles.com/local--files/doc%3Acsg/CSG.pdf


XRT Renderer : simpler? way of handling CSG trees
---------------------------------------------------

* http://xrt.wikidot.com/search:site/q/csg 


Kensler state tables corrected in below page...

* http://xrt.wikidot.com/doc:csg


* http://xrt.wikidot.com/downloads

XRT appears to be provided only as a Windows binary 


How to map the above described algorithm to OptiX ?
-----------------------------------------------------




Selector
~~~~~~~~~~

A selector is similar to a group in that it is a collection of higher level
graph nodes. The number of nodes in the collection is set by
rtSelectorSetChildCount, and the individual children are assigned with
rtSelectorSetChild. Valid child types are rtGroup, rtGeometryGroup,
rtTransform, and rtSelector.  The main difference between selectors and groups
is that selectors do not have an acceleration structure associated with them.
Instead, a visit program is specified with rtSelectorSetVisitProgram. This
program is executed every time a ray encounters the selector node during graph
traversal. The program specifies which children the ray should continue
traversal through by calling rtIntersectChild.  A typical use case for a
selector is dynamic (i.e. per-ray) level of detail: an object in the scene may
be represented by a number of geometry nodes, each containing a different level
of detail version of the object. The geometry groups containing these different
representations can be assigned as children of a selector. The visit program
can select which child to intersect using any criterion (e.g. based on the
footprint or length of the current ray), and ignore the others.  As for groups
and other graph nodes, child nodes of a selector can be shared with other graph
nodes to allow flexible instancing.


Intersection
~~~~~~~~~~~~~~~~

Ray traversal invokes an intersection program when the current ray encounters
one of a Geometry object’s primitives. It is the responsibility of an
intersection program to compute whether the ray intersects with the primitive,
and to report the parametric t-value of the intersection. Additionally, the
intersection program is responsible for computing and reporting any details of
the intersection, such as surface normal vectors, through attribute variables.
Once the intersection program has determined the t-value of a ray-primitive
intersection, it must report the result by calling a pair of OptiX functions,
rtPotentialIntersection and rtReportIntersection.

::

    ￼__device__ bool rtPotentialIntersection( float tmin )
    ￼__device__ bool rtReportIntersection( unsigned int material )


rtPotentialIntersection 
    takes the intersection’s t-value as an argument. 
    If the t-value could potentially be the closest intersection of the current traversal 
    the function narrows the t-interval of the current ray accordingly and returns true. 
    If the t-value lies outside the t-interval the function returns false, 
    whereupon the intersection program may trivially return.

    If rtPotentialIntersection returns true, 
    the intersection program may then set any attribute variable values 
    and call rtReportIntersection. This function takes an unsigned int specifying 
    the index of a material that must be associated with an any hit and closest hit program. 
    This material index can be used to support primitives of several different 
    materials flattened into a single Geometry object. 
    Traversal then immediately invokes the corresponding any hit program. 
    Should that any hit program invalidate the intersection via the rtIgnoreIntersection function, 
    then rtReportIntersection will return false. Otherwise, it will return true.


partitioned intersect
~~~~~~~~~~~~~~~~~~~~~~~

This assumes the primitive is chopped into single basis shape subparts.

Loops over all the sub-parts of the primitive invoking 
rtPotentialIntersection/rtReportIntersection multiple times... 
leaving the task narrowing down to find the closest intersect tmin to OptiX

::

    1243 RT_PROGRAM void intersect(int primIdx)
    1244 {
    1245   const uint4& solid    = solidBuffer[primIdx];
    1246   unsigned int numParts = solid.y ;
    1247 
    1248   //const uint4& identity = identityBuffer[primIdx] ; 
    1249   //const uint4 identity = identityBuffer[instance_index*primitive_count+primIdx] ;  // just primIdx for non-instanced
    1250 
    1251   // try with just one identity per-instance 
    1252   uint4 identity = identityBuffer[instance_index] ;
    1253 
    1254 
    1255   for(unsigned int p=0 ; p < numParts ; p++)
    1256   {
    1257       unsigned int partIdx = solid.x + p ;
    1258 
    1259       quad q0, q1, q2, q3 ;
    1260 
    1261       q0.f = partBuffer[4*partIdx+0];
    1262       q1.f = partBuffer[4*partIdx+1];
    1263       q2.f = partBuffer[4*partIdx+2] ;
    1264       q3.f = partBuffer[4*partIdx+3];
    1265 
    1266       identity.z = q1.u.z ;  // boundary from partBuffer (see ggeo-/GPmt)
    1267 
    1268       int partType = q2.i.w ;
    1269 
    1270       // TODO: use enum
    1271       switch(partType)
    1272       {
    1273           case 0:
    1274                 intersect_aabb(q2, q3, identity);
    1275                 break ;
    1276           case 1:
    1277                 intersect_zsphere<false>(q0,q1,q2,q3,identity);
    1278                 break ;
    1279           case 2:
    1280                 intersect_ztubs(q0,q1,q2,q3,identity);
    1281                 break ;



How would a boolean_intersect look ? 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~






IceSL
------

* https://members.loria.fr/Sylvain.Lefebvre/icesl/


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
