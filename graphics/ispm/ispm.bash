# === func-gen- : graphics/ispm/ispm fgp graphics/ispm/ispm.bash fgn ispm fgh graphics/ispm
ispm-src(){      echo graphics/ispm/ispm.bash ; }
ispm-source(){   echo ${BASH_SOURCE:-$(env-home)/$(ispm-src)} ; }
ispm-vi(){       vi $(ispm-source) ; }
ispm-env(){      elocal- ; }
ispm-usage(){ cat << EOU

ISPM : Image Space Photon Mapping
===================================

Basis of technique stems from observation:

First bounce from a point light to geometry and last bounce 
to the eye (pinhole camera) share the feature of a single point 
of projection. This makes them amenable to to a well known 
highly optimized technique : rasterization, where 
3D geometry can be projected onto a 2D image plane.


Refs
-----

Hardware-Accelerated Global Illumination by Image Space Photon Mapping

* http://graphics.cs.williams.edu/papers/PhotonHPG09/
* http://graphics.cs.williams.edu/papers/PhotonHPG09/ISPM-HPG09-video.mp4

* http://s09.idav.ucdavis.edu/talks/11-Luebke-NVIDIA-BPS-case-study-siggraph2009.pdf

* http://ispm.philipshield.com
* http://bpeers.com/blog/?itemid=541



Related
--------

Fast Global Illumination Approximations on Deep G-Buffers

* http://graphics.cs.williams.edu/papers/DeepGBuffer14/

* http://graphics.cs.williams.edu


Taking ISPM further
----------------------

Combining Soft Shadow and Image Space Photon Mapping for Global Illumination.
Ren-Hao Yao, NCTU Masters Thesis, Moved to GPU using OptiX (no source)

* http://ir.nctu.edu.tw/bitstream/11536/48429/1/750801.pdf


Good alternate description of technique

* http://www3.cs.stonybrook.edu/~nuahmed/geekish/projects/grad/RTPMproject.pdf


Lingo
-------

Deferred Shading
       https://en.wikipedia.org/wiki/Deferred_shading

       First pass gathers data needed for subsequent shading into a geometry buffer
       as a series of textures.

       Known as a screen space shading technique, as OpenGL is rasterizing the 
       results of the pipeline into a 2D "image" plane.

       One key disadvantage of deferred rendering is the inability to handle transparency


G-Buffers
       Geometry buffer






Traditional Photon Mapping (eg Jensen book)
---------------------------------------------

* collects photons **incident** on diffuse surfaces 
* constructs a kd tree to enable quick nearest neighbour queries allowing 
  an estimate of indirect light coming from any point  
* at render time queries are made, which **gather** the photons 


SIGRAPH 2003 course on Monte Carlo Ray Tracing

* http://www.cs.rutgers.edu/~decarlo/readings/mcrt-sg03c.pdf



ISPM
-----

* per-photon technique 
* construct "photon volumes", rasterize "photon volumes" onto 2D space 


ISPM Observations from source
-------------------------------

Fragment shaders in *.pix write to gl_FragData[0..4] this is apparently 
a way for a shader to write to multiple buffers. As opposed to the 
old single buffer gl_FragColor.  Both are deprecated in modern OpenGL, which 
just adopts flexible naming for same thing ?

* http://stackoverflow.com/questions/19075125/does-gl-fragcolor-do-anything-that-gl-fragdata-does-not


Photon Mapping Background
--------------------------

* :google:`nvidia optix photon mapping`
* http://www.cs.mtu.edu/~shene/PUBLICATIONS/2005/photon.pdf
* http://web.cs.wpi.edu/~emmanuel/courses/cs563/write_ups/zackw/photon_mapping/PhotonMapping.html


EOU
}
ispm-dir(){ echo $(local-base)/env/graphics/ispm/ispm  ; }
ispm-cd(){  cd $(ispm-dir); }
ispm-mate(){ mate $(ispm-dir) ; }
ispm-get(){
   local dir=$(dirname $(ispm-dir)) &&  mkdir -p $dir && cd $dir
   local url=http://graphics.cs.williams.edu/papers/PhotonHPG09/ispm.zip
   local zip=$(basename $url)
   local nam=${zip/.zip}
   [ ! -f "$zip" ] && curl -L -O $url
   [ ! -d "$nam" ] && unzip -l $zip 
   [ ! -d "$nam" ] && unzip    $zip 


}
