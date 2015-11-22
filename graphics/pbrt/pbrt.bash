# === func-gen- : graphics/pbrt/pbrt fgp graphics/pbrt/pbrt.bash fgn pbrt fgh graphics/pbrt
pbrt-src(){      echo graphics/pbrt/pbrt.bash ; }
pbrt-source(){   echo ${BASH_SOURCE:-$(env-home)/$(pbrt-src)} ; }
pbrt-vi(){       vi $(pbrt-source) ; }
pbrt-env(){      elocal- ; }
pbrt-usage(){ cat << EOU

Physically Based Ray Tracing 
=============================

* http://www.pbrt.org/
* https://github.com/mmp/pbrt-v2
* https://github.com/mmp/pbrt-v3


Chapter PDFs
--------------

* http://www.sciencedirect.com/science/book/9780123750792


Spectral Rendering
--------------------

* http://www.cmlab.csie.ntu.edu.tw/~jasonlai/project/rendering/index.html

* http://psychopath.io/spectral-rendering/

* http://www.anyhere.com/gward/papers/PicturePerfect.pdf

* http://jo.dreggn.org/home/2014_herowavelength.pdf

* http://www.cs.virginia.edu/~cab6fh/bib/teach/linear_color_reps.pdf

  Linear Color Representations for Full Spectral Rendering

* http://pjreddie.com/media/files/Redmon_Thesis.pdf

  * https://github.com/pjreddie/RayTracer/blob/master/RayTracer.cpp
  * Spectrum class
  * CIE XYZ weighting functions covered briefly, but seems not used in the code, 
    instead r,g,b gaussians


Photon Mapping
---------------

* http://www.cs.cmu.edu/afs/cs/academic/class/15462-f14/www/lec_slides/a1-jensen.pdf



 


EOU
}
pbrt-dir(){ echo $(local-base)/env/graphics/pbrt/pbrt-v3 ; }
pbrt-cd(){  cd $(pbrt-dir); }
pbrt-get(){
   local dir=$(dirname $(pbrt-dir)) &&  mkdir -p $dir && cd $dir

   git clone https://github.com/mmp/pbrt-v3

}
