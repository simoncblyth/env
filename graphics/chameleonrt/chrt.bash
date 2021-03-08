# === func-gen- : graphics/chameleonrt/chrt fgp graphics/chameleonrt/chrt.bash fgn chrt fgh graphics/chameleonrt src base/func.bash
chrt-source(){   echo ${BASH_SOURCE} ; }
chrt-edir(){ echo $(dirname $(chrt-source)) ; }
chrt-ecd(){  cd $(chrt-edir); }
chrt-dir(){  echo $LOCAL_BASE/env/graphics/chameleonrt/chrt ; }
chrt-cd(){   cd $(chrt-dir); }
chrt-vi(){   vi $(chrt-source) ; }
chrt-env(){  elocal- ; }
chrt-usage(){ cat << EOU

ChameleonRT
=============

* see also rt-


An example path tracer that runs on multiple ray tracing backends
(Embree/DXR/OptiX/Vulkan/Metal/OSPRay)


* https://github.com/Twinklebear/ChameleonRT

* https://www.willusher.io/graphics/2020/12/20/rt-dive-m1

  Compaurison of Metal, OptiX 7, VulkanRT, DXR 




* https://www.willusher.io


EOU
}
chrt-get(){
   local dir=$(dirname $(chrt-dir)) &&  mkdir -p $dir && cd $dir

}
