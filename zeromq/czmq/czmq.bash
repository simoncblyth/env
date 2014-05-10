# === func-gen- : zeromq/czmq/czmq fgp zeromq/czmq/czmq.bash fgn czmq fgh zeromq/czmq
czmq-src(){      echo zeromq/czmq/czmq.bash ; }
czmq-source(){   echo ${BASH_SOURCE:-$(env-home)/$(czmq-src)} ; }
czmq-vi(){       vi $(czmq-source) ; }
czmq-env(){      elocal- ; }
czmq-usage(){ cat << EOU

CZMQ : High-level C Binding for 0MQ
=====================================

* http://czmq.zeromq.org
* http://czmq.zeromq.org/page:get-the-software
* https://github.com/zeromq/czmq

* https://github.com/zeromq/pyczmq

Dependencies
-------------

* libzmq
* optionally : uuid-devel pkg


EOU
}
czmq-fold(){ echo $(local-base)/env/zeromq/czmq ; }
czmq-dir(){ echo $(czmq-fold)/$(czmq-name) ; }
czmq-sdir(){ echo $(env-home)/zeromq/czmq ; }

czmq-bindir(){ echo $(local-base)/env/bin ; }

czmq-cd(){  cd $(czmq-dir); }
czmq-scd(){  cd $(czmq-sdir); }
czmq-mate(){ mate $(czmq-dir) ; }
czmq-name(){  echo czmq-2.0.3 ; }
czmq-url(){ echo http://download.zeromq.org/$(czmq-name).tar.gz ; }
czmq-get(){
   local dir=$(dirname $(czmq-dir)) &&  mkdir -p $dir && cd $dir
   local url=$(czmq-url)
   local tgz=$(basename $url)
   local nam=${tgz/.tar.gz}
   [ ! -f "$tgz" ] && curl -L -O $url
   [ ! -d "$nam" ] && tar zxvf $tgz
}

czmq-prefix(){   echo $(czmq-fold) ; }
czmq-configure(){
   czmq-cd
   zeromq-
   ./configure --with-libzmq=$(zeromq-prefix) --prefix=$(czmq-prefix)
}
czmq-make(){
   czmq-cd
   make  $*
}
czmq-install(){ czmq-make install ; }
czmq-ls(){
   ls $(czmq-prefix)/include $(czmq-prefix)/lib
}

czmq-bin(){ echo $(czmq-bindir)/$1 ; }
czmq-cc(){
   local name=$1
   local bin=$(czmq-bin $name)
   mkdir -p $(dirname $bin)
   echo $msg compiling $bin 
   zeromq-
   cc $name.c -o $bin \
         -I$(zeromq-prefix)/include \
         -I$(czmq-prefix)/include \
         -L$(zeromq-prefix)/lib -lzmq \
         -L$(czmq-prefix)/lib -lczmq \
         -Wl,-rpath=$(czmq-prefix)/lib \
         -Wl,-rpath=$(zeromq-prefix)/lib
}




