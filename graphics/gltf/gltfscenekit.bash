# === func-gen- : graphics/gltf/gltfscenekit fgp graphics/gltf/gltfscenekit.bash fgn gltfscenekit fgh graphics/gltf
gltfscenekit-src(){      echo graphics/gltf/gltfscenekit.bash ; }
gltfscenekit-source(){   echo ${BASH_SOURCE:-$(env-home)/$(gltfscenekit-src)} ; }
gltfscenekit-vi(){       vi $(gltfscenekit-source) ; }
gltfscenekit-env(){      elocal- ; }
gltfscenekit-usage(){ cat << EOU

glTF loader for SceneKit
==========================

* https://github.com/magicien/GLTFSceneKit



EOU
}
gltfscenekit-dir(){ echo $(local-base)/env/graphics/gltf/GLTFSceneKit ; }
gltfscenekit-cd(){  cd $(gltfscenekit-dir); }
gltfscenekit-mate(){ mate $(gltfscenekit-dir) ; }
gltfscenekit-get(){
   local dir=$(dirname $(gltfscenekit-dir)) &&  mkdir -p $dir && cd $dir

   [ ! -d GLTFSceneKit ] && git clone git@github.com:magicien/GLTFSceneKit.git


}
