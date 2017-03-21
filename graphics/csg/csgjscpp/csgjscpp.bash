# === func-gen- : graphics/csg/csgjscpp/csgjscpp fgp graphics/csg/csgjscpp/csgjscpp.bash fgn csgjscpp fgh graphics/csg/csgjscpp
csgjscpp-src(){      echo graphics/csg/csgjscpp/csgjscpp.bash ; }
csgjscpp-source(){   echo ${BASH_SOURCE:-$(env-home)/$(csgjscpp-src)} ; }
csgjscpp-vi(){       vi $(csgjscpp-source) ; }
csgjscpp-env(){      elocal- ; }
csgjscpp-usage(){ cat << EOU





EOU
}
csgjscpp-dir(){ echo $(local-base)/env/graphics/csg/csgjs-cpp ; }
csgjscpp-cd(){  cd $(csgjscpp-dir); }
csgjscpp-mate(){ mate $(csgjscpp-dir) ; }
csgjscpp-get(){
   local dir=$(dirname $(csgjscpp-dir)) &&  mkdir -p $dir && cd $dir


    git clone https://github.com/dabroz/csgjs-cpp

}
