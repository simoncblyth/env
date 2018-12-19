# === func-gen- : graphics/pipeline/pipeline fgp graphics/pipeline/pipeline.bash fgn pipeline fgh graphics/pipeline src base/func.bash
pipeline-source(){   echo ${BASH_SOURCE} ; }
pipeline-edir(){ echo $(dirname $(pipeline-source)) ; }
pipeline-ecd(){  cd $(pipeline-edir); }
pipeline-dir(){  echo $LOCAL_BASE/env/graphics/pipeline/pipeline ; }
pipeline-cd(){   cd $(env-home)/graphics/pipeline ;  }
pipeline-vi(){   vi $(pipeline-source) ; }
pipeline-env(){  elocal- ; }
pipeline-usage(){ cat << EOU

Graphics Pipeline
====================

Favorite Computer Science Course on 3D graphics pipeline : Wolfgang Hurst, Utrecht University
------------------------------------------------------------------------------------------------

NB some of the earlier lectures are clearer than the 2013 ones detailed below

* https://www.youtube.com/channel/UCC6nn11Ozyz8ow91mf6RxFA
* https://www.youtube.com/playlist?list=PLbCDZQXIq7uYaf263gr-zb0wZGoCL-T5G


Introduction to Computer Graphics 2013, Wolfgang Hurst, Utrecht University
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Linear and Affine Transforms 

* https://www.youtube.com/watch?v=4I2S5Xhf24o&index=10&list=PLbCDZQXIq7uYaf263gr-zb0wZGoCL-T5G&t=51s

Computer Graphics 2013, Lect. 5(2) - Linear and affine transformations

* https://www.youtube.com/watch?v=KAW7lXxMSb4&list=PLbCDZQXIq7uYaf263gr-zb0wZGoCL-T5G&index=10

Computer Graphics 2013, Lect. 7(1) - Pipeline: perspective projection

* https://www.youtube.com/watch?v=Rixtn9_WzzU&index=13&list=PLbCDZQXIq7uYaf263gr-zb0wZGoCL-T5G

* ~13:00 View frustum 
* ~14:30 World Space to camera space transform (as its easier to put the camera at origin)
* 19:01 pipeline : World space -> camera space -> Orthographic view volume -> canonical view volume -> screen space
* 22:00 mapping the canonical view volume (-1:1 cube) into pixel grid 

::

        nx/2 0     0  nx/2-1/2      (pixel origin at bottom left)
        0    ny/2  0  ny/2-1/2
        0    0     1  1              drag the z along, need 3d for the rest if pipeline
        0    0     0  1              (need z info, even although dont draw it)

* 26:00  orthographic view volume, box with corners (r,t,f) (l,b,n) (r,b,n) 
         to canonical box with corners (-1,-1,-1) (1,1,1) etc...

1st translate center to origin::

        1   0   0    -(l+r)/2
        0   1   0    -(b+t)/2
        0   0   1    -(n+f)/2
        0   0   0      1

Scaling::

       2/(r-l) 0        0         0
       0       2/(t-b)  0         0
       0       0        2/(n-f)   0
       0       0        0         1

Combine those::

       2/(r-l) 0        0         -(r+l)/(r-l)
       0       2/(t-b)  0         -(t+b)/(t-b)
       0       0        2/(n-f)   -(n+f)/(n-f)
       0       0        0         1


* 29:10 world space (x,y,z) to camera space (eye,gaze)  u,v,w  u:gaze v:left w:up
* 33:00 up vector
* 34:06 aligning coordinate systems : origins then basis vectors

Origins : subtract the eye vector::
 
      1      0         0       -xe
      0      1         0       -ye
      0      0         1       -ze
      0      0         0        1

* 36:30 base vectors, matrix with colums the base vectors of camera frame is the rotation matrix::  

      xu     yu        zu      0    *   the above 
      xv     yv        zv      0
      xw     yw        zw      0
      0      0         0       1


Computer Graphics 2013, Lect. 7(2) - Pipeline: perspective projection
------------------------------------------------------------------------

* https://www.youtube.com/watch?v=mQKIn1oZ7Fg&list=PLbCDZQXIq7uYaf263gr-zb0wZGoCL-T5G&index=14

* 0:40 View frustum to orthographic view volume 

* 6:00 perspective projection aint really correct, it just preserves z-order, the z values
       are not strictly correct : due to the limitation of doing it with matrix multiplication  


* 14:22 extend homogenous (x,y,z,1) -> (x,y,z,w)   where eventually do (x/w,y/w,z/w, 1) 
* 17:15

Looking in -ve z, projection plane at near plane::

      n   0    0    0
      0   n    0    0
      0   0    n+f  -fn
      0   0    1    0

      x
      y
      z
      1

      x n/n
      y n/n
      [z(n+f)-f]/n 
      z/n 

      Dividing by "w" z/n, homogenize : the perspective divide

      nx/z          <-- z=n -> x
      ny/z          <-- z=n -> y 
      (n+f)-fn/z    <-- z=f -> f, z=n -> n 
      1 

* 24:48  view frustum transformation

* see ~/env/graphics/pipeline/pipeline.py for sympy Matrices 

* 34:56 Overview, Mper could also be Morth 

     x_pixel         =     Mvp(view-port) Mperspective Mcam   x 
     y_pixel                                                  y
     z_canonical                                              z
     1                                                        1


*  World Space
*  Camera Space   (eye position, gaze vector, up vector)
*  Orthographic View Volume
*  Canonical View Volume
*  Screen Pixels


Computer Graphics 2011, Lect. 7(1) - Perspective projection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* 08:43 World Space : cartesian coordinate system (x,y,z)
* 10:19 Camera position  : eye, gaze, fov
* 11:06 View frustum :  VF : l,r,n,f,t,b
* 13:23 Camera transformation : convention look in -Z direction in camera frame, X to right, Y up

* 16:13 Orthographic View Volume OVV : box containing same objects as frustum 
        (constructed such that an orthographic projection of the OVV matches 
         what a perspective projection of the view frustum VF would give)

        A special squeeze of the VF pyramid into a box that preserves sizes and 
        adhers to above relation 

* 17:53 Canonical View Volume : CVV : -1:1 cube, origin at center
* 18:45 Windowing transform : -1:1 -> nx,ny



Computer Graphics 2012, Lect. 7(2) - Perspective Projection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Slides
---------


* http://www.cs.uu.nl/docs/vakken/gr/2018/slides/lecture10_v2.pdf
* ~/opticks_refs/utrecht_cs_graphics_pipeline_lecture10_v2.pdf

* http://www.cs.uu.nl/docs/vakken/gr/2012-13/index.html
* http://www.cs.uu.nl/docs/vakken/gr/2012-13/gr_lectures.html

* http://www.cs.uu.nl/docs/vakken/gr/2012-13/Slides/INFOGR_2012-2013_lecture-07_projection.pdf
* ~/opticks_refs/INFOGR_2012-2013_lecture-07_projection.pdf

* https://vimeo.com/67642381




EOU
}
pipeline-get(){
   local dir=$(dirname $(pipeline-dir)) &&  mkdir -p $dir && cd $dir

}
