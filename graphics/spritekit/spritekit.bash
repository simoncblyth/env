# === func-gen- : graphics/spritekit/spritekit fgp graphics/spritekit/spritekit.bash fgn spritekit fgh graphics/spritekit
spritekit-src(){      echo graphics/spritekit/spritekit.bash ; }
spritekit-source(){   echo ${BASH_SOURCE:-$(env-home)/$(spritekit-src)} ; }
spritekit-vi(){       vi $(spritekit-source) ; }
spritekit-env(){      elocal- ; }
spritekit-usage(){ cat << EOU


* https://www.raywenderlich.com/96822/sprite-kit-tutorial-drag-drop-sprites


EOU
}
spritekit-dir(){ echo $(local-base)/env/graphics/spritekit/graphics/spritekit-spritekit ; }
spritekit-cd(){  cd $(spritekit-dir); }
spritekit-mate(){ mate $(spritekit-dir) ; }
spritekit-get(){
   local dir=$(dirname $(spritekit-dir)) &&  mkdir -p $dir && cd $dir

}
