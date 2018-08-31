# === func-gen- : graphics/gltf/viewer/bwasty fgp graphics/gltf/viewer/bwasty.bash fgn bwasty fgh graphics/gltf/viewer src base/func.bash
bwasty-source(){   echo ${BASH_SOURCE} ; }
bwasty-edir(){ echo $(dirname $(bwasty-source)) ; }
bwasty-ecd(){  cd $(bwasty-edir); }
bwasty-dir(){  echo $LOCAL_BASE/env/graphics/gltf/viewer/bwasty/gltf-viewer ; }
bwasty-cd(){   cd $(bwasty-dir); }
bwasty-vi(){   vi $(bwasty-source) ; }
bwasty-env(){  elocal- ; }
bwasty-usage(){ cat << EOU


glTF 2.0 Viewer written in Rust
==================================

* https://github.com/bwasty/gltf-viewer



EOU
}
bwasty-get(){
   local dir=$(dirname $(bwasty-dir)) &&  mkdir -p $dir && cd $dir

   git clone https://github.com/bwasty/gltf-viewer.git
   cd gltf-viewer
   cargo install gltf-viewer


}
