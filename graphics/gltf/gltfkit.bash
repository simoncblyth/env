# === func-gen- : graphics/gltf/gltfkit fgp graphics/gltf/gltfkit.bash fgn gltfkit fgh graphics/gltf
gltfkit-src(){      echo graphics/gltf/gltfkit.bash ; }
gltfkit-source(){   echo ${BASH_SOURCE:-$(env-home)/$(gltfkit-src)} ; }
gltfkit-vi(){       vi $(gltfkit-source) ; }
gltfkit-env(){      elocal- ; }
gltfkit-usage(){ cat << EOU

https://github.com/warrenm/GLTFKit



EOU
}
gltfkit-dir(){ echo $(local-base)/env/graphics/gltf/gltfkit ; }
gltfkit-cd(){  cd $(gltfkit-dir); }
gltfkit-mate(){ mate $(gltfkit-dir) ; }
gltfkit-get(){
   local dir=$(dirname $(gltfkit-dir)) &&  mkdir -p $dir && cd $dir


   [ ! -d gtlfkit ] && git clone https://github.com/warrenm/GLTFKit 

}
