# === func-gen- : graphics/csg/csgformat/csgformat fgp graphics/csg/csgformat/csgformat.bash fgn csgformat fgh graphics/csg/csgformat
csgformat-src(){      echo graphics/csg/csgformat/csgformat.bash ; }
csgformat-source(){   echo ${BASH_SOURCE:-$(env-home)/$(csgformat-src)} ; }
csgformat-vi(){       vi $(csgformat-source) ; }
csgformat-env(){      elocal- ; }
csgformat-usage(){ cat << EOU





EOU
}
csgformat-dir(){ echo $(local-base)/env/graphics/csg/csg-format ; }
csgformat-cd(){  cd $(csgformat-dir); }
csgformat-mate(){ mate $(csgformat-dir) ; }
csgformat-get(){
   local dir=$(dirname $(csgformat-dir)) &&  mkdir -p $dir && cd $dir

   [ ! -d "csg-format" ] && git clone https://github.com/megaton/csg-format

}
