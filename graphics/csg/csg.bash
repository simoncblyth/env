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



Geant4 CSG
-----------

::

   g4-cls G4UnionSolid
   g4-cls G4BooleanSolid 
   g4-cls G4VSolid


CSG logic picking which distance to which constituent done in eg G4UnionSolid::

    097     G4double DistanceToIn( const G4ThreeVector& p,
     98                            const G4ThreeVector& v  ) const ;
     99 
    100     G4double DistanceToIn( const G4ThreeVector& p ) const ;
    101 
    102     G4double DistanceToOut( const G4ThreeVector& p,
    103                             const G4ThreeVector& v,
    104                             const G4bool calcNorm=false,
    105                                   G4bool *validNorm=0,
    106                                   G4ThreeVector *n=0 ) const ;
    107 
    108     G4double DistanceToOut( const G4ThreeVector& p ) const ;


Pure virtuals in base G4VSolid::

    119     virtual EInside Inside(const G4ThreeVector& p) const = 0;
    120       // Returns kOutside if the point at offset p is outside the shapes
    121       // boundaries plus Tolerance/2, kSurface if the point is <= Tolerance/2
    122       // from a surface, otherwise kInside.
    123 
    124     virtual G4ThreeVector SurfaceNormal(const G4ThreeVector& p) const = 0;
    125       // Returns the outwards pointing unit normal of the shape for the
    126       // surface closest to the point at offset p.
    127 
    128     virtual G4double DistanceToIn(const G4ThreeVector& p,
    129                                   const G4ThreeVector& v) const = 0;
    130       // Return the distance along the normalised vector v to the shape,
    131       // from the point at offset p. If there is no intersection, return
    132       // kInfinity. The first intersection resulting from `leaving' a
    133       // surface/volume is discarded. Hence, it is tolerant of points on
    134       // the surface of the shape.
    135 
    136     virtual G4double DistanceToIn(const G4ThreeVector& p) const = 0;
    137       // Calculate the distance to the nearest surface of a shape from an
    138       // outside point. The distance can be an underestimate.
    139 
    140     virtual G4double DistanceToOut(const G4ThreeVector& p,
    141                    const G4ThreeVector& v,
    142                    const G4bool calcNorm=false,
    143                    G4bool *validNorm=0,
    144                    G4ThreeVector *n=0) const = 0;
    145       // Return the distance along the normalised vector v to the shape,
    146       // from a point at an offset p inside or on the surface of the shape.
    147       // Intersections with surfaces, when the point is < Tolerance/2 from a
    148       // surface must be ignored.
    149       // If calcNorm==true:
    150       //    validNorm set true if the solid lies entirely behind or on the
    151       //              exiting surface.
    152       //    n set to exiting outwards normal vector (undefined Magnitude).
    153       //    validNorm set to false if the solid does not lie entirely behind
    154       //              or on the exiting surface
    155       // If calcNorm==false:
    156       //    validNorm and n are unused.
    157       //
    158       // Must be called as solid.DistanceToOut(p,v) or by specifying all
    159       // the parameters.
    160 
    161     virtual G4double DistanceToOut(const G4ThreeVector& p) const = 0;
    162       // Calculate the distance to the nearest surface of a shape from an
    163       // inside point. The distance can be an underestimate.
    164 





github CSG
------------

* https://github.com/jtramm/ConstructiveSolidGeometry.jl
* https://github.com/jtramm/ConstructiveSolidGeometry.jl/blob/master/examples/1-Introduction.ipynb


Embree CSG : Computer Science Thesis describing Embree CSG
-------------------------------------------------------------

* https://dspace.cvut.cz/bitstream/handle/10467/65282/F3-DP-2016-Karaffova-Marketa-Efektivni_sledovani_paprsku_v_CSG_modelech.pdf?sequence=-1
* ~/opticks_refs/F3-DP-2016-Karaffova-Marketa-Efektivni_sledovani_paprsku_v_CSG_modelech.pdf


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



CSG OptiX 
--------------------------------------------------------



Constructive solid geometry
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://devtalk.nvidia.com/default/topic/771034/?comment=4296423

sphere/box intersection, nljones:

Your ray payload needs to contain a bit that indicates whether the ray is in a
sphere. Set it to one upon entering the sphere and zero upon leaving. 

## sign(N.D) determines if entering/leaving the shape

Your closest hit program for the sphere sends a new ray in the same direction with
this bit set. Your closest hit program for the box sends a new ray in the same
direction if the bit is one and sets the color of the ray payload if the bit is
zero.

In order to render the interface between the box and shere where they touch,
you also need to keep a bit indicating whether the ray is inside the box.


Example code for CSG in OptiX
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://devtalk.nvidia.com/default/topic/967816/?comment=4985663

dlacewell:

.. haven't thought about this too much, but for a limited number of closed
shapes, you could use per ray data (PRD) to store a hit counter, or really just
a flag, for each shape. 

## bitfield in per-ray-data with 1 or 2 bits for each basis shape could handle
## boolean operations involving small numbers of shapes (as is usual in G4 geometries)  

Use the closest hit program to either terminate the ray
and shade, or toggle the hit flag in PRD for the current shape and continue the
ray using rtTrace. 
Terminate when all hit flags are toggled on at once, meaning
that the current point is inside all the shapes.

##  rtTerminateRay only available in AnyHit, so by this dlacewell presumably means 
##  that can either accept a closest hit when per-ray-data flags are as they should
##  be for the boolean expression being evaluated OR if not (when this is not a real surface)
##  can call rtTrace again (from a modified starting position ? or tmin ) 

For some shapes, you could use the geometric normals to determine whether the
ray is entering or exiting, and then you might not need hit flags.

It may also be possible to do this with an any-hit program for a very small
number of shapes, by storing all intersections for the ray and sorting/shading
them in the ray gen program. That would be slow if there were too many shapes.


You could take optixSpherePP in the SDK and make some changes:

* add another sphere to the scene, that uses the same material
* add a geometry id variable to the sphere, and expose it as an attribute for the closest hit program
* change the closest hit program to make one of the spheres completely 
  transparent based on id, and continue the ray with rtTrace. 
* Make the spheres semi-transparent. You still shoot a new ray with rtTrace, 
  but composite the result with the current sphere color and opacity using per ray data.

Once you get that all working, then it's probably not a big jump to CSG.







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
