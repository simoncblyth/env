# === func-gen- : graphics/isosurface/dualcontouring fgp graphics/isosurface/dualcontouring.bash fgn dualcontouring fgh graphics/isosurface
dualcontouring-src(){      echo graphics/isosurface/dualcontouring.bash ; }
dualcontouring-source(){   echo ${BASH_SOURCE:-$(env-home)/$(dualcontouring-src)} ; }
dualcontouring-vi(){       vi $(dualcontouring-source) ; }
dualcontouring-env(){      elocal- ; }
dualcontouring-usage(){ cat << EOU


Dual Contouring Implementation in C++ 
========================================

**The use of all code is limited to non-profit research purposes only.**

Author: Tao Ju (with QEF code written by Scott Schaefer)
Updated: February 2011

Input Formats
--------------

Both formats store an octree grid with
inside/outside signs. 

dcf 
    Dual Contouring Format contains intersection points and normals on grid

sog 
    Signed Octree with Geometry contains a single point location within each non-empty 
    grid cell. 

Both formats can be produced from a polygonal model, via scan-conversion,
using the Polymender software on my website: 


* http://www1.cse.wustl.edu/~taoju/code/polymender.htm

The detail formats are documented in the readme file of Polymender.

Two algorithms are implemented in this code, switch in main() in dc.cpp

* original dual contouring algorithm [Ju et al., Siggraph 2002] 
* intersection-free extension [Ju et al., Pacific Graphics 2006]. 

You can switch between them in the main() function in dc.cpp. 
In addition, octree simplification (guided by QEF errors) 
is also implemented, and can be turned on in the main() function.


EOU
}

dualcontouring-nam(){ echo dc ; }
dualcontouring-url(){ echo https://downloads.sourceforge.net/project/dualcontouring/dc_cpp.zip ; }

dualcontouring-dir(){ echo $(local-base)/env/graphics/isosurface/dualcontouring/$(dualcontouring-nam) ; }
dualcontouring-cd(){  cd $(dualcontouring-dir); }
dualcontouring-get(){
   local dir=$(dirname $(dualcontouring-dir)) &&  mkdir -p $dir && cd $dir

   local url=$(dualcontouring-url)
   local nam=$(dualcontouring-nam)
   local dst=$(basename $url)

   [ ! -f $dst ] && curl -L -O $url 
   [ ! -d $nam ] && unzip $dst 



}
