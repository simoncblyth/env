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


Cubic
--------


* https://www.researchgate.net/publication/220389759_Solving_cubics_by_polynomial_fitting
* ~/opticks_refs/Strobach_cubic_root_fitting.pdf


* ~/opticks_refs/polynomial_scaling_0727007.pdf
* http://epubs.siam.org/doi/abs/10.1137/0727007




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

Cubic
-------

* http://mathworld.wolfram.com/CubicFormula.html
* ~/opticks_refs/Cubic_Formula_Wolfram_MathWorld.pdf

* http://pages.mtu.edu/~tbco/cm3230/Real_Roots_of_Cubic_Equation.pdf
* ~/opticks_refs/Real_Roots_of_Cubic_Equation.pdf


* ~/opticks_refs/nickalls_cubic1993.pdf
* http://www.nickalls.org/dick/papers/maths/cubic1993.pdf

* ~/opticks_refs/nickalls_quartic2009.pdf
* http://www.nickalls.org/dick/papers/maths/quartic2009.pdf


opencv solveCubic
~~~~~~~~~~~~~~~~~~~

* http://docs.opencv.org/3.0-beta/search.html?q=solveCubic&check_keywords=yes&area=default
* https://github.com/opencv/opencv/blob/26be2402a3ad6c9eacf7ba7ab2dfb111206fffbb/modules/core/src/mathfuncs.cpp


CUDA Complex
--------------------

* #include <cuComplex.h>

CUDA Floating Point
--------------------

* ~/opticks_refs/NVIDIA-CUDA-Floating-Point.pdf
* http://developer.download.nvidia.com/assets/cuda/files/NVIDIA-CUDA-Floating-Point.pdf

Poly34
--------

* http://math.ivanovo.ac.ru/dalgebra/Khashin/poly/index.html
* https://brownmath.com/alge/polysol.htm


Neumark
---------

* :google:`quartic neumark`
 

Don Herbison-Evans
-------------------

Test of a quartic solving routine, 24 June 1994

* http://www.realtimerendering.com/resources/GraphicsGems/gemsv/ch1-1/quarcube.c

* http://index-of.co.uk/Game-Development/Programming/Graphics%20Gems%205.pdf
* ~/opticks_refs/Graphics_Gems_5.pdf

* ~/opticks_refs/Graphics_Gems_5_QuarticCubic.pdf




Eberly : approach using Rational type
---------------------------------------

* https://www.geometrictools.com/Documentation/LowDegreePolynomialRoots.pdf


Roots of low order polynomials, Terence R.F.Nonweiler
--------------------------------------------------------

* http://dl.acm.org/citation.cfm?id=363039

P51
------

* http://iowahills.com/Downloads/Iowa%20Hills%20P51%20Root%20Finder.zip




g4-cls G4AnalyticalPolSolver
-----------------------------


* this appears not to be used for G4Torus, the iterative Jenkins-Traub is used 

::


     29 // Class description:
     30 //
     31 // G4AnalyticalPolSolver allows the user to solve analytically a polynomial
     32 // equation up to the 4th order. This is used by CSG solid tracking functions
     33 // like G4Torus.
     34 //
     35 // The algorithm has been adapted from the CACM Algorithm 326:
     36 //
     37 //   Roots of low order polynomials
     38 //   Author: Terence R.F.Nonweiler
     39 //   CACM  (Apr 1968) p269
     40 //   Translated into C and programmed by M.Dow
     41 //   ANUSF, Australian National University, Canberra, Australia
     42 //   m.dow@anu.edu.au
     43 //
     44 // Suite of procedures for finding the (complex) roots of the quadratic,
     45 // cubic or quartic polynomials by explicit algebraic methods.
     46 // Each Returns:
     47 //
     48 //   x=r[1][k] + i r[2][k]  k=1,...,n, where n={2,3,4}
     49 //
     50 // as roots of:
     51 // sum_{k=0:n} p[k] x^(n-k) = 0
     52 // Assumes p[0] != 0. (< or > 0) (overflows otherwise)
     53 
     54 // --------------------------- HISTORY --------------------------------------
     55 //
     56 // 13.05.05 V.Grichine ( Vladimir.Grichine@cern.ch )
     57 //          First implementation in C++


     64 class G4AnalyticalPolSolver
     65 {
     66   public:  // with description
     67 
     68     G4AnalyticalPolSolver();
     69     ~G4AnalyticalPolSolver();
     70 
     71     G4int QuadRoots(    G4double p[5], G4double r[3][5]);
     72     G4int CubicRoots(   G4double p[5], G4double r[3][5]);
     73     G4int BiquadRoots(  G4double p[5], G4double r[3][5]);
     74     G4int QuarticRoots( G4double p[5], G4double r[3][5]);
     75 };



P51
----

* http://iowahills.com/P51RootFinder.html

* Roots of Low Order Polynomials" by Terence R.F.Nonweiler CACM

* http://www.apc.univ-paris7.fr/~franco/g4doxy/html/G4AnalyticalPolSolver_8hh-source.html


Numerics
----------

* https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html

The expression x^2 - y^2 is another formula that exhibits catastrophic cancellation. 
It is more accurate to evaluate it as (x - y)(x + y).


Why sqrt(epsilon) ?
~~~~~~~~~~~~~~~~~~~~~

* https://scicomp.stackexchange.com/questions/14355/choosing-epsilons

* https://en.wikipedia.org/wiki/Numerical_differentiation#Practical_considerations_using_floating_point_arithmetic

  * Numerical differenciation: f(x+h)-f(x)   suggests to use h=sqrt(eps)*x for x!=0

* http://www.maths.manchester.ac.uk/~higham/talks/asna13_cardiff.pdf


Refs
------

* https://archive.org/details/SolutionOfCubicQuarticEquations
* https://ia802707.us.archive.org/32/items/SolutionOfCubicQuarticEquations/Neumark-SolutionOfCubicQuarticEquations.pdf
* ~/opticks_refs/Neumark-SolutionOfCubicQuarticEquations.pdf

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

quartic-p51(){

   local dir=$(dirname $(quartic-dir)) &&  mkdir -p $dir && cd $dir
   local url="http://iowahills.com/Downloads/Iowa%20Hills%20P51%20Root%20Finder.zip"

   [ ! -f p51.zip ] && curl -L -o p51.zip "$url"
   [ ! -d p51 ] && unzip p51.zip -d p51  
   
   cd p51


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

