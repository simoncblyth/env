# === func-gen- : vr/vrworks/vrworks fgp vr/vrworks/vrworks.bash fgn vrworks fgh vr/vrworks
vrworks-src(){      echo vr/vrworks/vrworks.bash ; }
vrworks-source(){   echo ${BASH_SOURCE:-$(env-home)/$(vrworks-src)} ; }
vrworks-vi(){       vi $(vrworks-source) ; }
vrworks-env(){      elocal- ; }
vrworks-usage(){ cat << EOU

VRWORKS : NVIDIAs SDK for VR
===============================

Pascal 
-------


* http://www.roadtovr.com/nvidia-explains-pascal-simultaneous-multi-projection-lens-matched-shading-for-vr/
* https://blogs.nvidia.com/blog/2016/05/06/pascal-vrworks/





EOU
}
vrworks-dir(){ echo $(local-base)/env/vr/vrworks/vr/vrworks-vrworks ; }
vrworks-cd(){  cd $(vrworks-dir); }
vrworks-mate(){ mate $(vrworks-dir) ; }
vrworks-get(){
   local dir=$(dirname $(vrworks-dir)) &&  mkdir -p $dir && cd $dir

}
