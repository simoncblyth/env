# === func-gen- : graphics/amd/radeonrays fgp graphics/amd/radeonrays.bash fgn radeonrays fgh graphics/amd src base/func.bash
radeonrays-source(){   echo ${BASH_SOURCE} ; }
radeonrays-edir(){ echo $(dirname $(radeonrays-source)) ; }
radeonrays-ecd(){  cd $(radeonrays-edir); }
radeonrays-dir(){  echo $LOCAL_BASE/env/graphics/amd/radeonrays ; }
radeonrays-cd(){   cd $(radeonrays-dir); }
radeonrays-vi(){   vi $(radeonrays-source) ; }
radeonrays-env(){  elocal- ; }
radeonrays-usage(){ cat << EOU

RadeonRays 4.0
-----------------

* https://github.com/GPUOpen-LibrariesAndSDKs/RadeonRays_SDK
* https://gpuopen.com/radeon-rays/

Below link to documentation is broken:

* https://radeon-pro.github.io/RadeonProRenderDocs/rr/about.html

The below works:

* https://radeon-pro.github.io/RadeonProRenderDocs/en/index.html
* https://radeon-pro.github.io/RadeonProRenderDocs/en/rr/about.html
* https://radeon-pro.github.io/RadeonProRenderDocs/en/rr/data_structs/rrgeometrybuildinput.html


Looks like it is just a triangles only system.

* https://www.pcgamer.com/amd-rdna2-release-date-big-navi-specs-price-performance/

  AMD RDNA 2 GPU, late 2020 release with hardware ray tracing  




EOU
}
radeonrays-get(){
   local dir=$(dirname $(radeonrays-dir)) &&  mkdir -p $dir && cd $dir

}
