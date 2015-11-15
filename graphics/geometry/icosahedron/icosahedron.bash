# === func-gen- : graphics/geometry/icosahedron/icosahedron fgp graphics/geometry/icosahedron/icosahedron.bash fgn icosahedron fgh graphics/geometry/icosahedron
icosahedron-src(){      echo graphics/geometry/icosahedron/icosahedron.bash ; }
icosahedron-source(){   echo ${BASH_SOURCE:-$(env-home)/$(icosahedron-src)} ; }
icosahedron-vi(){       vi $(icosahedron-source) ; }
icosahedron-env(){      elocal- ; }
icosahedron-usage(){ cat << EOU





EOU
}
icosahedron-dir(){ echo $(env-home)/graphics/geometry/icosahedron ; }
icosahedron-cd(){  cd $(icosahedron-dir); }
icosahedron-get(){
   local dir=$(dirname $(icosahedron-dir)) &&  mkdir -p $dir && cd $dir

}
