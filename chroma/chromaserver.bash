# === func-gen- : chroma/chromaserver fgp chroma/chromaserver.bash fgn chromaserver fgh chroma
chromaserver-src(){      echo chroma/chromaserver.bash ; }
chromaserver-source(){   echo ${BASH_SOURCE:-$(env-home)/$(chromaserver-src)} ; }
chromaserver-vi(){       vi $(chromaserver-source) ; }
chromaserver-env(){      elocal- ; }
chromaserver-usage(){ cat << EOU

CHROMA SERVER
==============

https://github.com/mastbaum/chroma-server

This appears to now be integrated with chroma implemented
in chroma/bin/chroma-server


ZMQ
----

::

    sudo port install zmq   # zeromq-3.2.3



EOU
}
chromaserver-dir(){ echo $(local-base)/env/chroma/chroma-server ; }
chromaserver-cd(){  cd $(chromaserver-dir); }
chromaserver-mate(){ mate $(chromaserver-dir) ; }
chromaserver-get(){
   local dir=$(dirname $(chromaserver-dir)) &&  mkdir -p $dir && cd $dir

   git clone https://github.com/mastbaum/chroma-server

}
