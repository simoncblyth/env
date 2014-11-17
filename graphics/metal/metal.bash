# === func-gen- : graphics/metal/metal fgp graphics/metal/metal.bash fgn metal fgh graphics/metal
metal-src(){      echo graphics/metal/metal.bash ; }
metal-source(){   echo ${BASH_SOURCE:-$(env-home)/$(metal-src)} ; }
metal-vi(){       vi $(metal-source) ; }
metal-env(){      elocal- ; }
metal-usage(){ cat << EOU

Metal : close to the metal OpenGL ES alternative on iOS
=========================================================

* http://metalbyexample.com/



EOU
}
metal-dir(){ echo $(local-base)/env/graphics/metal/graphics/metal-metal ; }
metal-cd(){  cd $(metal-dir); }
metal-mate(){ mate $(metal-dir) ; }
metal-get(){
   local dir=$(dirname $(metal-dir)) &&  mkdir -p $dir && cd $dir

}
