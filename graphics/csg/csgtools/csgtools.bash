# === func-gen- : graphics/csg/csgtools/csgtools fgp graphics/csg/csgtools/csgtools.bash fgn csgtools fgh graphics/csg/csgtools
csgtools-src(){      echo graphics/csg/csgtools/csgtools.bash ; }
csgtools-source(){   echo ${BASH_SOURCE:-$(env-home)/$(csgtools-src)} ; }
csgtools-vi(){       vi $(csgtools-source) ; }
csgtools-env(){      elocal- ; }
csgtools-usage(){ cat << EOU





EOU
}
csgtools-dir(){ echo $(local-base)/env/graphics/csg/csg-tools ; }
csgtools-cd(){  cd $(csgtools-dir); }
csgtools-mate(){ mate $(csgtools-dir) ; }
csgtools-get(){
   local dir=$(dirname $(csgtools-dir)) &&  mkdir -p $dir && cd $dir

   [ ! -d "csg-tools" ] && git clone https://github.com/megaton/csg-tools

}
