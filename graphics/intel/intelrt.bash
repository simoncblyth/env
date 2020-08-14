# === func-gen- : graphics/intel/intelrt fgp graphics/intel/intelrt.bash fgn intelrt fgh graphics/intel src base/func.bash
intelrt-source(){   echo ${BASH_SOURCE} ; }
intelrt-edir(){ echo $(dirname $(intelrt-source)) ; }
intelrt-ecd(){  cd $(intelrt-edir); }
intelrt-dir(){  echo $LOCAL_BASE/env/graphics/intel/intelrt ; }
intelrt-cd(){   cd $(intelrt-dir); }
intelrt-vi(){   vi $(intelrt-source) ; }
intelrt-env(){  elocal- ; }
intelrt-usage(){ cat << EOU

Intel Rendering Framework
==========================


* https://itpeernetwork.intel.com/intel-rendering-framework-xe-architecture/#gs.csgpe3

I’m pleased to share today that the Intel® Xe architecture roadmap for data
center optimized rendering includes ray tracing hardware acceleration support
for the Intel® Rendering Framework family of API’s and libraries.

See Also
----------

* embree-



EOU
}
intelrt-get(){
   local dir=$(dirname $(intelrt-dir)) &&  mkdir -p $dir && cd $dir

}
