# === func-gen- : graphics/metal/metal fgp graphics/metal/metal.bash fgn metal fgh graphics/metal
metal-src(){      echo graphics/metal/metal.bash ; }
metal-source(){   echo ${BASH_SOURCE:-$(env-home)/$(metal-src)} ; }
metal-vi(){       vi $(metal-source) ; }
metal-env(){      elocal- ; }
metal-usage(){ cat << EOU

Metal : close to the metal OpenGL ES alternative on iOS and OSX
=================================================================


Warren Moore
-------------

* http://metalbyexample.com/
* https://github.com/metal-by-example
* https://realm.io/news/3d-graphics-metal-swift/
* https://github.com/warrenm/slug-swift-metal/

Amund Tveit
-------------

* http://memkite.com/blog/2015/06/10/swift-and-metal-gpu-programming-on-osx-10-11-el-capitan/
* http://memkite.com/blog/2014/12/30/example-of-sharing-memory-between-gpu-and-cpu-with-swift-and-metal-for-ios8/
* https://github.com/atveit

Simon Gladman
--------------

* https://realm.io/news/swift-summit-simon-gladman-metal/
* https://github.com/FlexMonkey
* https://github.com/FlexMonkey/MetalKit-Particles
* https://github.com/FlexMonkey/ParticleLab





EOU
}
metal-dir(){ echo $(local-base)/env/graphics/metal/graphics/metal-metal ; }
metal-cd(){  cd $(metal-dir); }
metal-mate(){ mate $(metal-dir) ; }
metal-get(){
   local dir=$(dirname $(metal-dir)) &&  mkdir -p $dir && cd $dir

}
