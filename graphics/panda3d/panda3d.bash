# === func-gen- : graphics/panda3d/panda3d fgp graphics/panda3d/panda3d.bash fgn panda3d fgh graphics/panda3d
panda3d-src(){      echo graphics/panda3d/panda3d.bash ; }
panda3d-source(){   echo ${BASH_SOURCE:-$(env-home)/$(panda3d-src)} ; }
panda3d-vi(){       vi $(panda3d-source) ; }
panda3d-env(){      elocal- ; }
panda3d-usage(){ cat << EOU


* http://www.panda3d.org/download.php?platform=macosx&version=1.7.2&sdk
* https://developer.nvidia.com/cg-toolkit

EOU
}
panda3d-dir(){ echo $(local-base)/env/graphics/panda3d/graphics/panda3d-panda3d ; }
panda3d-cd(){  cd $(panda3d-dir); }
panda3d-mate(){ mate $(panda3d-dir) ; }
panda3d-get(){
   local dir=$(dirname $(panda3d-dir)) &&  mkdir -p $dir && cd $dir


   http://www.panda3d.org/download.php?platform=macosx&version=1.7.2&sdk

}
