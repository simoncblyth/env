# === func-gen- : graphics/intersect/intersect fgp graphics/intersect/intersect.bash fgn intersect fgh graphics/intersect
intersect-src(){      echo graphics/intersect/intersect.bash ; }
intersect-source(){   echo ${BASH_SOURCE:-$(env-home)/$(intersect-src)} ; }
intersect-vi(){       vi $(intersect-source) ; }
intersect-env(){      elocal- ; }
intersect-usage(){ cat << EOU

INTERSECT : ray geometry intersection notes
=========================================================

A place for referencing sources and comparing implementations


See Also
---------

Moved index of precursors to opticksdev-

* csg- covering basic refs and serialization 
* isosurface- for extracting polygons from CSG trees
* tboolean- testing Opticks implementations of raytraced and polygonized CSG trees
* sdf- signed distance functions, used to CPU side geometry modelling 
* scene- scene description languages

Exceptional Sources
---------------------

Big table linking to varions intersection imps

* http://www.realtimerendering.com/intersections.html
* http://geomalgorithms.com/algorithms.html

* :google:`erit intersect`
* ~/opticks_refs/erit_intersection_collection.pdf 



Many (includine surface-of-revolution), terse:

* http://hugi.scene.org/online/hugi24/coding%20graphics%20chris%20dragan%20raytracing%20shapes.htm

Torus
--------

* ~/opticks_refs/don_cross_cosinekitty_raytrace_a4.pdf
* http://www.cosinekitty.com/raytrace/chapter13_torus.html

  * ~/opticks_refs/cosinekitty_The_Torus_class.pdf
  * direct solve quartic

* http://users.wowway.com/~phkahler/torus.pdf


* https://github.com/erich666/GraphicsGems/blob/master/gemsii/intersect/inttor.c


* https://github.com/erich666/GraphicsGems

Quartic
~~~~~~~~~

Moved to quartic-

Skala
~~~~~~~~~

* http://www.wseas.org/multimedia/journals/computers/2013/025705-201.pdf
* ~/opticks_refs/skala_torus_intersect_025705-201.pdf

* https://pdfs.semanticscholar.org/a781/e21f47aae7f72ce4434fe6a360afe92b0e93.pdf
* ~/opticks_refs/skala_torus_line_bounding.pdf


Intersect equivalent to that between a line and swept sphere

* also rotational symmetry means could rotate the line to make a cone, and
  intersect the cone with the sphere  

Ruminations
~~~~~~~~~~~~~~~


Use cone/sphere intersect to skip solving quartic:

* testing cone (from rotating line about torus axis) 
  against sphere (that would sweep to form torus) 

* http://mathworld.wolfram.com/Cone-SphereIntersection.html

  * perhaps no simplification when intersect, but gives way of classifying roots,
    to skip solving quartic when possible

* http://www.flipcode.com/archives/Frustum_Culling.shtml



Sphere Cone
~~~~~~~~~~~~~~~~

* ~/opticks_refs/eberly_IntersectionSphereCone.pdf
* https://www.geometrictools.com/Documentation/IntersectionSphereCone.pdf

* https://www.gamedev.net/forums/topic/555628-sphere-cone-test-with-no-sqrt-for-frustum-culling/




Using squared inequalities, this in turn leads to the following sqrt-free formulation:

::

    V = sphere.center - cone.apex_location
    // use cone apex frame 

    a = dotProduct(V, cone.direction_normal)
    // distance of sphere along cone axis

    p = a * cone_sin
    q = cone_cos * cone_cos * dotProduct(V, V) - a * a
    r = q - sphere_radius * sphere_radius
    if (p<sphere_radius) || (q>0):
         if (r < 2 * sphere_radius * p), the sphere is partially included (return -1)
         else if q<0, the sphere is totally included (return 1)
         else cull the sphere (return 0)
    else:
        if ( -r < 2 * sphere_radius * p), the sphere is partially included (return -1)
        else if q<0, the sphere is totally included (return 1)
        else cull the sphere (return 0)




* https://bartwronski.com/2017/04/13/cull-that-cone/


* https://gist.github.com/jcayzac/1241840

V = sphere.center - cone.apex_location
a = V * cone.direction_normal
b = a * cone.tan
c = sqrt( V*V - a*a )
d = c - b
e = d * cone.cos

now  if ( e >= sphere.radius ) , cull the sphere
else if ( e <=-sphere.radius ) , totally include the sphere
else the sphere is partially included.

What's going on in this cone-sphere test? Basically, I'm trying to find 'e'
which is the shortest distance from the center of the sphere to the surface of
the cone. You can draw some pictures and see what's going on. 'a' is how far
along the ray of the cone the sphere's center is. 'b' is the radius of the cone
at 'a'. 'c' is the distance from the center of the sphere to the axis of the
cone, and 'd' is the distance from the center of the sphere to the surface of
the cone, along a line perpendicular to the axis of the cone (which is not the
closest distance).

Note that once you compute 'a' you could tell that the sphere intersects the
cone just by testing

    Square( a ) <= V*V * Square( cone.cos )

This is very fast and nice, but it can't tell us if the sphere was partially included or totally included, so it's no good for heirarchy.


:google:`Charles Bloom algorithm for sphere-cone intersection`




Algebraic
~~~~~~~~~~~~~

* 
* ~/opticks_refs/ray_tracing_algebraic_surfaces_p83-hanrahan.pdf




AlexanderTolmachev/ray-tracer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://github.com/AlexanderTolmachev/ray-tracer/blob/master/src/torus.cpp


Thesis : Torus and Simple Surface Intersection Based on a Configuration Space Approach
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* http://bh.knu.ac.kr/~kujinkim/kjkim_thesis.pdf
* ~/opticks_refs/torus_intersect_kjkim_thesis.pdf

p7 image showing three cases of torus, r < R, r = R, r > R  (only interested in r < R )

p31 torus-cone intersect



:google:`ray trace torus`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* http://steve.hollasch.net/cgindex/render/raytorus.html

* http://hugi.scene.org/online/hugi24/coding%20graphics%20chris%20dragan%20raytracing%20shapes.htm



g4-cls G4UTorus
~~~~~~~~~~~~~~~~~

* vecgeom-;vecgeom-cls TorusImplementation2


g4-cls G4Torus
~~~~~~~~~~~~~~~~~

::

     254 // Calculate the real roots to torus surface. 
     255 // Returns negative solutions as well.
     256 
     257 void G4Torus::TorusRootsJT( const G4ThreeVector& p,
     258                             const G4ThreeVector& v,
     259                                   G4double r,
     260                                   std::vector<G4double>& roots ) const
     261 {
     262 
     263   G4int i, num ;
     264   G4double c[5], srd[4], si[4] ;
     265 
     266   G4double Rtor2 = fRtor*fRtor, r2 = r*r  ;
     267 
     268   G4double pDotV = p.x()*v.x() + p.y()*v.y() + p.z()*v.z() ;
     269   G4double pRad2 = p.x()*p.x() + p.y()*p.y() + p.z()*p.z() ;
     270 
     271   c[0] = 1.0 ;
     272   c[1] = 4*pDotV ;
     273   c[2] = 2*(pRad2 + 2*pDotV*pDotV - Rtor2 - r2 + 2*Rtor2*v.z()*v.z()) ;
     274   c[3] = 4*(pDotV*(pRad2 - Rtor2 - r2) + 2*Rtor2*p.z()*v.z()) ;
     275   c[4] = pRad2*pRad2 - 2*pRad2*(Rtor2+r2)
     276        + 4*Rtor2*p.z()*p.z() + (Rtor2-r2)*(Rtor2-r2) ;
     277 
     278   G4JTPolynomialSolver  torusEq;
     279 
     280   num = torusEq.FindRoots( c, 4, srd, si );
     281 
     282   for ( i = 0; i < num; i++ )
     283   {
     284     if( si[i] == 0. )  { roots.push_back(srd[i]) ; }  // store real roots
     285   }
     286 
     287   std::sort(roots.begin() , roots.end() ) ;  // sorting  with <
     288 }


Thoughts on partitioning, use if it simplifies
-------------------------------------------------

* OptiX primitives do not need to correspond to input solids...
  eg experience with analytic PMT

* Polygonization into triangles is partitioning taken to the extreme (with approximation thrown in), 
  accel structures can efficiently cope with millions of triangles so do not 
  be coy about partitioning solids if that allows a simpler intersection 
  implementation.

  For example polycones could be partitioned at the z-planes leaving cone 
  intersections within a z range. Internally this is what the polycone 
  implementations do anyhow, treating them as separate primitives just shifts 
  the burden of locating which of the sub-polycone to OptiX accel structures 
  rather than the polycone implementation.

* fly in ointment of partitioning is boolean CSG, if the shape takes 
  part in boolean relations then cannot partition 
  because the boolean handling is done
  at intersect level beneath the OptiX primitive level


Line Cone
------------

* https://www.geometrictools.com/Documentation/IntersectionLineCone.pdf
* ~/opticks_refs/Eberly_IntersectionLineCone.pdf 

* https://www.csie.ntu.edu.tw/~cyy/courses/rendering/pbrt-2.00/html/cone_8cpp_source.html

  Quadratic approach

* http://lousodrome.net/blog/light/2017/01/03/intersection-of-a-ray-and-a-cone/

* http://mathworld.wolfram.com/Cone.html


Quadratic
~~~~~~~~~~~~~~

* https://everything2.com/title/Ray+Tracing%253A+intersections+with+cones

t2(yD2 + zD2 - xD2) + t(2yEyD + 2zEzD - 2xExD) + (yE2 + zE2 - xE2) = 0
which is simply a quadratic equation having roots:

where a = (yD2 + zD2 - xD2)
      b = (2yEyD + 2zEzD - 2xExD)
      c = (yE2 + zE2 - xE2)

Giving the intersections of the ray with the cone in the parameter t. To find
the position of the first intersection in x,y,z space we must substitute back
into either equation the value of t found at each root. Additionally, we must
now check that the x value found lies within the x range permitted above. The
first t root which lies in the allowed range in x space will give us the
required position. N.B. for cones with their base closer to the y-axis than
their point, we simply translate the other end of the double cone into that
region of the x-axis. 

**End Caps** 

Assume the end cap was at the xmin end, 
if one of the (real) x roots lies to the left of xmin and the other lies to the right of xmin, 
then we know that the ray must have passed through the end cap, 
and we can find the position at which it would have intersected 
the infinite plane normal to the cone and passing through xmin by

tend = xmin - xE
     ------------
          xD

and again convert t-space to x-space in the ray equation.
We should however consider the special case where the ray travels 
parallel to the axis of the cone. Here, the quadratic equation will 
have no real solutions (no cone point in x-range) or one real solution 
(the point of the cone), but will have intersected the end cap(s). 
Therefore, the ray tracer must be carefully coded to catch this case.







Ray Sphere, Geometrical and Algebraic Approaches
-------------------------------------------------

* http://kylehalladay.com/blog/tutorial/math/2013/12/24/Ray-Sphere-Intersection.html

* https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-sphere-intersection


Numerically stable quadratic solution
-----------------------------------------

* https://people.csail.mit.edu/bkph/articles/Quadratics.pdf
* http://www.it.uom.gr/teaching/linearalgebra/NumericalRecipiesInC/c5-6.pdf




Surface of Revolution
-----------------------

A surface which can be generated by revolving a plane curve about an axis in its plane.

Ray tracing books mention symmetry use to intersect in 2D 
(project ray into plane of the generatrix)



* https://en.wikipedia.org/wiki/Surface_of_revolution

* :google:`ray intersection with surface of revolution`

* http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=934677

  * An Old Problem with a new perspective

* https://www.mathworks.com/matlabcentral/answers/73606-intersection-of-3d-ray-and-surface-of-revolution


Normal to a quadric surface
-----------------------------

Implicit unit sphere

   F(x,y,z) = 0 ;    x^2 + y^2 + z^2 - 1 = 0 

   Normal at [df/dx, df/dy, df/dz ] = [2x, 2y, 2z]  normalized -> [x,y,z]



Is Polycone a piecewise surface of revolution ?
--------------------------------------------------

::

    G4Polycone(const G4String& pName,
                 G4double  phiStart,
                 G4double  phiTotal,
                 G4int         numZPlanes,
                 const G4double  zPlane[],
                 const G4double  rInner[],
                 const G4double  rOuter[])

    G4Polycone(const G4String& pName, 
                 G4double  phiStart,
                 G4double  phiTotal,
                 G4int         numRZ,
                 const G4double  r[],
                 const G4double  z[])



* http://geant4.in2p3.fr/IMG/pdf_Lecture-Geometry.pdf

  * shows a polycone with convex parts, 
    (r,z) coords not with monotonic z ?  or used rInner > 0 for those sections


Intersections ray/cone
-------------------------

* https://www.geometrictools.com/Documentation/IntersectionLineCone.pdf
* https://www.csie.ntu.edu.tw/~cyy/courses/rendering/pbrt-2.00/html/cone_8cpp_source.html


hemi-pmt.cu is poor name, much more general now
------------------------------------------------

* analytic.cu is better, however this needs to be sliced into focussed headers

::

    simon:cu blyth$ opticks-find hemi-pmt.cu 
    ./bin/oks.bash:    ./optixrap/cu/hemi-pmt.cu
    ./optixrap/cu/hemi-pmt.cu:  rtPrintf("hemi-pmt.cu:bounds primIdx %d min %10.4f %10.4f %10.4f max %10.4f %10.4f %10.4f \n", primIdx, 
    ./ggeo/GParts.cc:    // see oxrap/cu/hemi-pmt.cu::intersect
    ./ggeo/GParts.cc:    // following access pattern of oxrap/cu/hemi-pmt.cu::intersect
    ./ggeo/GPmt.cc:which are used in cu/hemi-pmt.cu as the OptiX primitives
    ./optixrap/OGeo.cc:    geometry->setIntersectionProgram(m_ocontext->createProgram("hemi-pmt.cu.ptx", "intersect"));
    ./optixrap/OGeo.cc:    geometry->setBoundingBoxProgram(m_ocontext->createProgram("hemi-pmt.cu.ptx", "bounds"));
    ./ggeo/GParts.hh:        // allowing this to copied/used on GPU in cu/hemi-pmt.cu
    ./opticksnpy/NPrism.cpp:    // hmm more dupe of hemi-pmt.cu/make_prism
    ./opticksnpy/NTrianglesNPY.cpp:    // hmm how to avoid duplication between here and hemi-pmt.cu/make_prism
    ./optixrap/CMakeLists.txt:    cu/hemi-pmt.cu 
    ./optixrap/CMakeLists.txt:    ${CMAKE_CURRENT_BINARY_DIR}/${name}_generated_hemi-pmt.cu.ptx
    ./ana/pmt/geom.py:        see cu/hemi-pmt.cu for where these are used 
    simon:opticks blyth$ 





EOU
}
intersect-dir(){ echo $(local-base)/env/graphics/intersect/graphics/intersect-intersect ; }
intersect-cd(){  cd $(intersect-dir); }
intersect-mate(){ mate $(intersect-dir) ; }
intersect-get(){
   local dir=$(dirname $(intersect-dir)) &&  mkdir -p $dir && cd $dir

}
