# === func-gen- : firerays/firerays fgp firerays/firerays.bash fgn firerays fgh firerays
firerays-src(){      echo firerays/firerays.bash ; }
firerays-source(){   echo ${BASH_SOURCE:-$(env-home)/$(firerays-src)} ; }
firerays-vi(){       vi $(firerays-source) ; }
firerays-env(){      elocal- ; }
firerays-usage(){ cat << EOU

Firerays
=========

* http://raytracey.blogspot.tw/2015/08/firerays-amds-opencl-based-high.html
* http://gpuopen.com/firerays-2-0-open-sourcing-and-customizing-ray-tracing/
* https://github.com/GPUOpen-LibrariesAndSDKs/RadeonRays_SDK


EOU
}
firerays-dir(){ echo $(local-base)/env/firerays/firerays-firerays ; }
firerays-cd(){  cd $(firerays-dir); }
firerays-mate(){ mate $(firerays-dir) ; }
firerays-get(){
   local dir=$(dirname $(firerays-dir)) &&  mkdir -p $dir && cd $dir

}
