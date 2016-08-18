# === func-gen- : graphics/gltf/gltf fgp graphics/gltf/gltf.bash fgn gltf fgh graphics/gltf
gltf-src(){      echo graphics/gltf/gltf.bash ; }
gltf-source(){   echo ${BASH_SOURCE:-$(env-home)/$(gltf-src)} ; }
gltf-vi(){       vi $(gltf-source) ; }
gltf-env(){      elocal- ; }
gltf-usage(){ cat << EOU

GLTF
=====

* https://www.khronos.org/news/press/khronos-collada-now-recognized-as-iso-standard
* https://github.com/KhronosGroup/glTF/blob/master/specification/README.md
* https://www.khronos.org/gltf
* https://github.com/KhronosGroup/glTF#gltf-tools



EOU
}
gltf-dir(){ echo $(local-base)/env/graphics/gltf/graphics/gltf-gltf ; }
gltf-cd(){  cd $(gltf-dir); }
gltf-mate(){ mate $(gltf-dir) ; }
gltf-get(){
   local dir=$(dirname $(gltf-dir)) &&  mkdir -p $dir && cd $dir

}
