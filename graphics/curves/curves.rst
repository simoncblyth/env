curves
=========

* https://gamedev.stackexchange.com/questions/129481/why-arent-regular-quadratic-and-cubic-splines-used-much-in-games

* https://www.redblobgames.com/articles/curved-paths/

* :google:`bezier curve around circle`

* https://spencermortensen.com/articles/bezier-circle/

Refs
-------

https://www.geeksforgeeks.org/difference-between-spline-b-spline-and-bezier-curves/


1. Spline :

A spline curve can be specified by giving a specified set of coordinate
positions, called control points which indicate the general shape of the curve.

2. B-Spline :

B-Spline is a basis function that contains a set of control points. The
B-Spline curves are specified by Bernstein basis function that has limited
flexibility.

3. Bezier :

These curves are specified with boundary conditions, with a characterizing
matrix or with blending function. A Bezier curve section can be filled by any
number of control points. The number of control points to be approximated and
their relative position determine the degree of Bezier polynomial.


Catmull-Rom Spline 
----------------------

Programming & Using Splines - Part#1

* https://www.youtube.com/@javidx9
* https://www.youtube.com/watch?v=9_aJGUTePYo
 
  Looped Catmull-Rom : ~19min 

For segment with 4 control points with indices [i-1,i,i+1,i+2] 
the spline curve goes through points with indices::

   i (t=0)
   i+1 (t=1) 


* https://lucidar.me/en/mathematics/catmull-rom-splines/



Cubic-Spline Interpolation
-----------------------------


* https://www.youtube.com/c/ProfJeffreyChasnov

Cubic Spline Interpolation (Part A) | Lecture 44 | Numerical Methods for Engineers

* https://www.youtube.com/watch?v=LaolbjAzZvg

Cubic Spline Interpolation (Part B) | Lecture 45 | Numerical Methods for Engineers

* https://www.youtube.com/watch?v=4VpE9Tbie14


Exploring Bezier And Spline Curves

* https://www.youtube.com/@rdfuhr
* https://www.youtube.com/watch?v=-aiErrvLRfE
* https://richardfuhr.neocities.org/BusyBCurves.html

  Interactive Web App : Bezier/Spline curve definition with movable control points  



optix curves
-------------

* https://forums.developer.nvidia.com/t/curves-performance-in-optix-7-1/142746/2


dhart

In case it helps make tests that are exactly 1:1, you can convert uniform cubic
B-splines to Bezier with a quick matrix multiply. (and vice-versa with the
inverse, of course).::

     1  4  1  0
     0  4  2  0
     0  2  4  0
     0  1  4  1     (and divide by 6) 

   

* :google:`optix curve bezier`


* https://github.com/MikaZeilstra/RaytracingDiffusionCurves
* https://repository.tudelft.nl/islandora/object/uuid:3e8e5679-5e05-4989-81ac-0c5569614597?collection=education
* ~/opticks_refs/MikaZeilstra_RaytracingDiffusionCurves.pdf

However, A lot of research has been done about how to efficiently intersect
cubic splines and Optix already implements such an algorithm for us. This
algorithm is most probably based on work by Reshetov [15]. And takes a set of
three-dimensional cubic B- spline curves as input and produces intersections at
any curve and/or at the closest curve for a given ray. To leverage the speed of
this algorithm the curves will not be approximated but will be intersected
directly.

[15] Alexander Reshetov. Exploiting budan-fourier and vincent’s theorems for
ray tracing 3d bezier curves. In Proceedings of High Performance Graphics,
HPG ’17, New York, NY, USA, 2017. Association for Computing Ma- chinery.

* https://research.nvidia.com/sites/default/files/pubs/2017-07_Exploiting-Budan-Fourier-and/HPG2017-Budan-Fourier.pdf

* ~/opticks_refs/HPG2017-Budan-Fourier.pdf




The second mismatch is somewhat more complex. Namely, it requires unclamped
b-splines. This is a slightly different primitive than the Bezier splines
which are the input of the algorithm of this paper. There are two ways to
alleviate this problem.

 
