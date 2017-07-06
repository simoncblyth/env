# === func-gen- : graphics/csg/csgbsp/csgbsp fgp graphics/csg/csgbsp/csgbsp.bash fgn csgbsp fgh graphics/csg/csgbsp
csgbsp-src(){      echo graphics/csg/csgbsp/csgbsp.bash ; }
csgbsp-source(){   echo ${BASH_SOURCE:-$(env-home)/$(csgbsp-src)} ; }
csgbsp-vi(){       vi $(csgbsp-source) ; }
csgbsp-env(){      elocal- ; }
csgbsp-usage(){ cat << EOU


* https://en.wikipedia.org/wiki/Binary_space_partitioning

Originally discounted CSG BSP due to prejudice against
it for producing horrible triangles... 
but hybrid implicit/parametric is running into sticky subdiv territory. 
So perhaps could combine CSG BSP with a remeshing step afterwards
to clean up the tris.  

* https://github.com/simoncblyth/csgjs-cpp
* https://github.com/dabroz/csgjs-cpp
* https://github.com/evanw/csg.js/



EOU
}

csgbsp-url(){ echo https://github.com/simoncblyth/csgjs-cpp ; }
csgbsp-dir(){ echo $(local-base)/env/graphics/csg/csgjs-cpp; }
csgbsp-nam(){ echo $(basename $(csgbsp-dir)) ; }

csgbsp-cd(){  cd $(csgbsp-dir); }
csgbsp-get(){
   local dir=$(dirname $(csgbsp-dir)) &&  mkdir -p $dir && cd $dir

   local url=$(csgbsp-url) 
   local nam=$(csgbsp-nam) 

   [ ! -f "$nam" ] && git clone $url 


}
