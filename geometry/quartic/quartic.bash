# === func-gen- : geometry/quartic/quartic fgp geometry/quartic/quartic.bash fgn quartic fgh geometry/quartic
quartic-src(){      echo geometry/quartic/quartic.bash ; }
quartic-source(){   echo ${BASH_SOURCE:-$(env-home)/$(quartic-src)} ; }
quartic-vi(){       vi $(quartic-source) ; }
quartic-env(){      elocal- ; }
quartic-usage(){ cat << EOU

Quartic
==========

NEXT
-----

* put together into opticks torus primitive, but in 
  flexible separated way to allow swapping out 
  the quartic imp ... torus/ray coeff prep can live in primitive 


Approaches Investigated
----------------------------

yairchu/quartic fork : flexi approach with normalization, reciprocation, stableness test...
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ported fork of yairchu/quartic to run on device almost just requiring CUDA_BOTH, 
  it operates ok in simple tests but requires increasing stack size as 
  uses recursion to handle degree sliding polys

  * seems not so easy to simplify 
  * can torus-ray gaurantee something about the coeffs ? To allow simplification 
  * does coeff shifting in code 

::

   quartic-c
   nvcc -arch=sm_30 quartic_real.cu -run ; rm a.out  


Roots3And4 from graphics gems : simple no robustness
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ported to CUDA, simpler : no robustness reversal or degree sliding

::

   quartic-c 
   nvcc -arch=sm_30 Roots3And4.cu -run ; rm a.out



AlexanderTolmachev/ray-tracer : simple no robustness
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://github.com/AlexanderTolmachev/ray-tracer/blob/2c29012fb36dfb3aff35c4761266c6841cd43205/lib/quarticsolver/src/quarticsolver.h
* https://github.com/AlexanderTolmachev/ray-tracer/blob/2c29012fb36dfb3aff35c4761266c6841cd43205/lib/quarticsolver/src/quarticsolver.cpp


vecgeom-;vecgeom-cls TorusImplementation2 : simple no robustness
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* SolveQuartic : looks simple



Refs
------


* https://www.jstor.org/stable/2975214
* ~/opticks_refs/quartic_geometric_faucette_2975214.pdf

* ~/opticks_refs/Wolfram_Quartic_Equation.pdf
* http://mathworld.wolfram.com/QuarticEquation.html

* http://www.sosmath.com/algebra/factor/fac12/fac12.html

* http://dl.acm.org/citation.cfm?id=2245297
* ~/opticks_refs/quartic-p94-zhao.pdf

* http://users.nik.uni-obuda.hu/sanyo/publications/fulltext/sami2015_submission_60.pdf
* ~/opticks_refs/quartic_sami2015_submission_60.pdf


* http://www.sciencedirect.com/science/article/pii/S0377042710002128
* ~/opticks_refs/strobach_fast_quartic.pdf


* http://www.ijpam.eu/contents/2011-71-2/7/7.pdf
* ~/opticks_refs/shmakov_universal_quartic.pdf

* Schwarze, Jochen, Cubic and Quartic Roots, Graphics Gems, p. 404-407, code: p. 738-786, Roots3And4.c.
* http://www.realtimerendering.com/resources/GraphicsGems/gems/Roots3And4.c

* ~/opticks_refs/quartics_cubics_for_graphics_tr487.pdf
* http://sydney.edu.au/engineering/it/research/tr/tr487.pdf

* ~/opticks_refs/quartic_solver-f90-a30-flocke.pdf
* http://dl.acm.org/citation.cfm?doid=2835205.2699468


* https://www.gamedev.net/forums/topic/451048-best-way-of-solving-a-polynomial-of-the-fourth-degree/

* :google:`github solve quartic`

* https://github.com/yairchu/quartic

* https://github.com/madbat/SwiftMath/blob/master/SwiftMath/Polynomial.swift

* https://en.wikipedia.org/wiki/Durand–Kerner_method


* http://icl.cs.utk.edu/lapack-forum/viewtopic.php?f=2&t=229

Peter (Strobach) is reporting than using close form formula to solve 
degree 4 polynomials ("quartic polynomials") is faster and more stable 
than DGEEV + companion matrix.



::

    In [6]: expand((x-1)*(x-2)*(x-3)*(x-4))
    Out[6]: x**4 - 10*x**3 + 35*x**2 - 50*x + 24

    simon:build blyth$ /tmp/env/lib/QuarticTest 1 -10 35 -50 24
    4.000000
    3.000000
    2.000000
    1.000000



EOU
}
quartic-prefix(){ echo /tmp/env ; }
quartic-dir(){ echo $(local-base)/env/geometry/quartic/quartic ; }

quartic-bdir(){   echo $(quartic-prefix)/build ; }
quartic-sdir(){   echo $(quartic-dir) ; }

quartic-c(){   cd $(quartic-dir); }
quartic-cd(){  cd $(quartic-dir); }
quartic-bcd(){  cd $(quartic-bdir) ; }
quartic-ecd(){  cd $(env-home)/geometry/quartic ; }
quartic-get(){
   local dir=$(dirname $(quartic-dir)) &&  mkdir -p $dir && cd $dir


   #[ ! -f Roots3And4.c ] && curl -L -O http://www.realtimerendering.com/resources/GraphicsGems/gems/Roots3And4.c

   [ ! -d quartic ] && git clone https://github.com/simoncblyth/quartic
}

quartic-cmake(){
   local iwd=$PWD
   local bdir=$(quartic-bdir)

   mkdir -p $bdir
   #[ -f "$bdir/CMakeCache.txt" ] && echo $msg configured already && return  

   quartic-bcd

   cmake \
       -DCMAKE_BUILD_TYPE=Debug \
       -DCMAKE_INSTALL_PREFIX=$(quartic-prefix) \
       $* \
       $(quartic-sdir)
}

quartic--()
{
    quartic-bcd 
    make ${1:-install}

}

