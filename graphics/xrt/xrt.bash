# === func-gen- : graphics/xrt/xrt fgp graphics/xrt/xrt.bash fgn xrt fgh graphics/xrt
xrt-src(){      echo graphics/xrt/xrt.bash ; }
xrt-source(){   echo ${BASH_SOURCE:-$(env-home)/$(xrt-src)} ; }
xrt-vi(){       vi $(xrt-source) ; }
xrt-env(){      elocal- ; }
xrt-usage(){ cat << EOU

XRT raytracer (Windows-only, no source)
==========================================

XRT is a raytracing based programmable rendering system for photo-realistic
image synthesis built around a plug-in architecture design.


* http://xrt.wdfiles.com/local--files/downloads/techref.pdf
* ~/opticks_refs/XRT_Renderer_techref.pdf

Includes C++ scene API based on the NVIDIA Gelato API (a predecessor to OptiX)
which looks a lot like renderman API.

* https://en.wikipedia.org/wiki/Gelato_(software)


Formulas for many implicit shapes
----------------------------------------

* http://xrt.wikidot.com/gallery:implicit




EOU
}
xrt-dir(){ echo $(local-base)/env/graphics/xrt/graphics/xrt-xrt ; }
xrt-cd(){  cd $(xrt-dir); }
xrt-mate(){ mate $(xrt-dir) ; }
xrt-get(){
   local dir=$(dirname $(xrt-dir)) &&  mkdir -p $dir && cd $dir

}
