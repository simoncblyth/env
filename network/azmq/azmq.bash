# === func-gen- : network/azmq/azmq fgp network/azmq/azmq.bash fgn azmq fgh network/azmq
azmq-src(){      echo network/azmq/azmq.bash ; }
azmq-source(){   echo ${BASH_SOURCE:-$(env-home)/$(azmq-src)} ; }
azmq-vi(){       vi $(azmq-source) ; }
azmq-env(){      elocal- ; }
azmq-usage(){ cat << EOU

AZMQ
======

C++ language binding library integrating ZeroMQ with Boost Asio

* https://github.com/zeromq/azmq


EOU
}
azmq-dir(){ echo $(local-base)/env/network/azmq ; }
azmq-cd(){  cd $(azmq-dir); }
azmq-get(){
   local dir=$(dirname $(azmq-dir)) &&  mkdir -p $dir && cd $dir
   [ ! -d azmq ] && git clone https://github.com/zeromq/azmq.git
}
