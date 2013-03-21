# === func-gen- : fossil/fossil fgp fossil/fossil.bash fgn fossil fgh fossil
fossil-src(){      echo fossil/fossil.bash ; }
fossil-source(){   echo ${BASH_SOURCE:-$(env-home)/$(fossil-src)} ; }
fossil-vi(){       vi $(fossil-source) ; }
fossil-env(){      elocal- ; }
fossil-usage(){ cat << EOU
Fossil
========

Simple, high-reliability, distributed software configuration management

  * http://www.fossil-scm.org/fossil/doc/trunk/www/index.wiki


EOU
}
fossil-nam(){ echo fossil-src-20130216000435 ; }
fossil-dir(){ echo $(local-base)/env/fossil/$(fossil-nam) ; }
fossil-cd(){  cd $(fossil-dir); }
fossil-mate(){ mate $(fossil-dir) ; }
fossil-get(){
   local dir=$(dirname $(fossil-dir)) &&  mkdir -p $dir && cd $dir
   local nam=$(fossil-nam)
   local tgz=$nam.tar.gz
   local url=http://www.fossil-scm.org/download/$tgz
   [ ! -f "$tgz" ] && curl -L -O $url
   [ ! -d "$nam" ] && tar zxvf $tgz
}

fossil-build(){
   fossil-cd
   mkdir -p build
   cd build
   ../configure
   make
}
fossil-bin(){ echo $(fossil-dir)/build/fossil ; }

fossil-install(){ [ ! -x $(env-home)/bin/fossil ] &&  ln -s $(fossil-bin) $(env-home)/bin/fossil ; }
   

