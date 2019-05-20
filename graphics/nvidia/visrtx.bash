# === func-gen- : graphics/nvidia/visrtx fgp graphics/nvidia/visrtx.bash fgn visrtx fgh graphics/nvidia src base/func.bash
visrtx-source(){   echo ${BASH_SOURCE} ; }
visrtx-edir(){ echo $(dirname $(visrtx-source)) ; }
visrtx-ecd(){  cd $(visrtx-edir); }
visrtx-dir(){  echo $LOCAL_BASE/env/graphics/nvidia/VisRTX ; }
visrtx-cd(){   cd $(visrtx-dir); }
visrtx-vi(){   vi $(visrtx-source) ; }
visrtx-env(){  elocal- ; }
visrtx-usage(){ cat << EOU

VisRTX : Visualization framework powered by NVIDIA RTX technology
=====================================================================

C++ rendering framework developed by the HPC Visualization Developer Technology team at NVIDIA

* https://github.com/NVIDIA/VisRTX
* https://gitmemory.com/tbiedert
* https://hpcvis.org/

* https://developer.nvidia.com/mdl-sdk

The MDL wrapper provided by VisRTX is self-contained and can be of interest to
anyone who wants to access the MDL SDK from an OptiX-based application.

Found this project by 

* :google:`optix DISABLE_ANYHIT`


EOU
}
visrtx-get(){
   local dir=$(dirname $(visrtx-dir)) &&  mkdir -p $dir && cd $dir

   [ ! -d VisRTX ] && git clone git@github.com:simoncblyth/VisRTX.git
}
