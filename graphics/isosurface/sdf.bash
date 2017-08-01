# === func-gen- : graphics/isosurface/sdf fgp graphics/isosurface/sdf.bash fgn sdf fgh graphics/isosurface
sdf-src(){      echo graphics/isosurface/sdf.bash ; }
sdf-source(){   echo ${BASH_SOURCE:-$(env-home)/$(sdf-src)} ; }
sdf-vi(){       vi $(sdf-source) ; }
sdf-env(){      elocal- ; }
sdf-usage(){ cat << EOU


SDF : Signed Distance Functions
==================================


Introductions
----------------

* http://www.alanzucconi.com/2016/07/01/signed-distance-functions/
* http://jamie-wong.com/2016/07/15/ray-marching-signed-distance-functions/


Lists of SDFs
--------------

* http://www.iquilezles.org/www/articles/distfunctions/distfunctions.htm
* http://mercury.sexy/hg_sdf/
* https://github.com/marklundin/glsl-sdf-primitives



cylinder Deltaphi segment with SDF ?
--------------------------------------

* CSG max to intersect cylinder with two segment planes thru origin


https://github.com/marklundin/glsl-sdf-primitives/blob/master/sdTriPrism.glsl

::

    float sdTriPrism( vec3 p, vec2 h )
    {
        vec3 q = abs(p);
        return max(q.z-h.y,max(q.x*0.866025+p.y*0.5,-p.y)-h.x*0.5);
    }



Bloomenthal on Implicit surfaces: understandable intro to math foundations
-----------------------------------------------------------------------------

* http://graphics.cs.northwestern.edu/~jet/Teach/2004_1winAdvGraphics/Papers/bloomenthal2002.pdf

Implicit function theorem parallels closed requirement for boolean **solids**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

From the implicit function theorem it may be shown that for f(p) = 0, where 0 a
regular value of f and f is continuous, the implicit surface is a
two-dimensional manifold [Bruce and Giblin 1992, prop. 4.16]. The
Jordan-Brouwer Separation Theorem states that such a manifold separates space
into the surface itself and two connected open sets: an infinite `outside' and
a finite ‘inside' [Guillemin and Pollack 1974].

Non-manifold Surfaces
~~~~~~~~~~~~~~~~~~~~~~~

* http://www.unchainedgeometry.com/jbloom/pdf/dis-chptr4-nonman.pdf

::

    As described in section 3.5.4, implicit surfaces are two-dimensional manifolds.
    That is, they envelope a volume. A manifold surface is, everywhere, locally
    homomorphic (that is, of comparable structure) to a two-dimensional disk
    (manifold surfaces with boundary are everywhere homomorphic to a disk or a
    half- disk). For example, a disk may be fully applied to any portion of the
    torus below, left. But, the disk does not fully apply to all points of a
    teapot, below, right. In particular, the disk is truncated along the upper
    boundary of the teapot bowl.


The contortions required to model non-manifold surfaces... discussed
in the above makes living with the **closed sub-object** "limitation" of CSG **solids**
seem a price well worth paying.



Grokking SDFs
---------------


Can SDFs model a bounded cylinder with no endcaps ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* :google:`signed distance function closed surface`

* are signed distance functions limited to closed surfaces ?


* https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3659211/

Indeed, in the codimension one case, there is a clearly defined interior and a
clearly defined exterior of the given object of interest, if the surface is
closed. In this case, a signed distance function can be used for a level set
implementation of the evolution equation.



Combining SDF with CSG operations
------------------------------------------


* complement(SDF) = -SDF                      # think of flipping the normals
* union(SDF_A,SDF_B) = min(SDF_A, SDF_B)      # think of distance to union of two overlapping spheres 
* intersect(SDF_A,SDF_B) = max(SDF_A, SDF_B)
* difference(SDF_A,SDF_B) = intersect(SDF_A, -SDF_B) = max(SDF_A, -SDF_B)

* negate to complemement
* difference is intersect with the complement

Scene defined by an SDF 

* http://jamie-wong.com/2016/07/15/ray-marching-signed-distance-functions/



half-space
~~~~~~~~~~~~~

::

                           +ve outside  
      - ------- sdf = z-h ----------   z= h 
                           -ve inside
      h    
      _ ____________________________ z = 0 


slab
~~~~~~

::
                           ~B  ~B   ~B
                             
      - ------- sdfA = z-h ----------   z = h 
                           
      h        A     A      ~B  ~B   ~B 
      _ ____________________________    z = 0 

               A     A      ~B  ~B   ~B

      - -------- sdfB = z+h ---------   z = -h

               A     A      B    B    B 

               A     A      B    B    B
 
               A     A      B    B    B


Slab is a difference of half-spaces

* sdfA = z - h      (plane at z = h) 
* sdfB = z + h      (plane at z = -h ),  
* ~sdfB = -(z+h)    (same position, but now inside are upwards to +z)

::

    intersect(sdfA, ~sdfB) 
    max( z - h , -(z + h) )
    max( z - h , -z - h )
    max(z, -z) - h
    abs(z) - h 



min/max/abs identities
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://gist.github.com/paniq/3f882c50f1790e323482


box
~~~~~

Box - signed - exact

::

    float sdBox( vec3 p, vec3 b )
    {
      vec3 d = abs(p) - b;  
      return min(max(d.x,max(d.y,d.z)),0.0) + length(max(d,0.0));
    }


::

    abs(p) ->  fold space into one "quadrant"

    vec3 d = abs(p) - b  (presumably b=box.max)
        ->  conveniently the origin planes now split inside/outside 

    when inside the box d.x,d.y,d.z will all be -ve so: max(d,0.0) -> 0 
    the larger of the -ve d.xyz will be the closest distance to the surface 
        min(max(d.x,max(d.y,d.z)),0.0)

    when outside the box, maximal d coord will be +ve so min(maximal, 0) 
    will get clamped at zero and the other term comes into play 

        length(max(d,0.0))


  Note the handling of two cases using min and max clamped to zero
       min( f(d), 0 ) + max( g(d), 0 )




Lots of shapes and modifiers
------------------------------

NVScene 2015 Session: How to Create Content with Signed Distance Functions (Johann Korndorfer) 

* https://m.youtube.com/watch?v=s8nFqwOho-s

  Amazing walkthrough of how very compact SDF code 
  using space modifiers can yield extensive geometry.

* http://mercury.sexy/hg_sdf/

  By Johann Korndorfer   

  Implementations of a lot of shapes, including platonics 
  and generalized distance functions based on 

  * https://www.viz.tamu.edu/faculty/ergun/research/implicitmodeling/papers/sm99.pdf


Rendering SDF : Ray marching, Sphere Tracing
------------------------------------------------

Sphere tracing:
a geometric method for the antialiased ray tracing of implicit surfaces

* http://graphics.cs.illinois.edu/sites/default/files/zeno.pdf
* ~/opticks_refs/Sphere_Tracing_John_C_Hart.pdf

* http://erleuchtet.org/~cupe/permanent/enhanced_sphere_tracing.pdf
* ~/opticks_refs/enhanced_sphere_tracing.pdf

* http://onlinepresent.org/proceedings/vol43_2013/32.pdf
* ~/opticks_refs/OptiX_SDF_Park.pdf 



What Is Ray Marching, Sphere Tracing ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://computergraphics.stackexchange.com/questions/161/what-is-ray-marching-is-sphere-tracing-the-same-thing


Intro Tutorial on graphics including Sphere Tracing, from a very beginner perspective
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://www.scratchapixel.com/index.php
* https://www.scratchapixel.com/lessons/advanced-rendering/rendering-distance-fields
* https://www.scratchapixel.com/lessons/advanced-rendering/rendering-distance-fields/maths-behind-sphere-tracing

Includes Lipschitz explanation.


Procedural modeling with signed distance functions (Thesis)
-------------------------------------------------------------

* http://aka-san.halcy.de/distance_fields_prefinal.pdf
* ~/opticks_refs/Procedural_Modelling_with_Signed_Distance_Functions_Thesis.pdf

* p25.. distance functions for sphere, torus, cylinder, cone, box



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
sdf-dir(){ echo $(local-base)/env/graphics/isosurface/sdf ; }
sdf-cd(){  cd $(sdf-dir); }

sdf-url(){ echo http://mercury.sexy/hg_sdf/hg_sdf.glsl ; }
sdf-nam(){ echo $(basename $(sdf-url)) ; } 
sdf-get(){
   local dir=$(sdf-dir) &&  mkdir -p $dir && cd $dir
   local url=$(sdf-url)
   local nam=$(sdf-nam)
   [ ! -f "$nam" ] && curl -L -O $url 
}

sdf-edit(){
    vi $(sdf-dir)/$(sdf-nam)

}

