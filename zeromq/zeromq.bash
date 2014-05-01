# === func-gen- : zeromq/zeromq fgp zeromq/zeromq.bash fgn zeromq fgh zeromq
zeromq-src(){      echo zeromq/zeromq.bash ; }
zeromq-source(){   echo ${BASH_SOURCE:-$(env-home)/$(zeromq-src)} ; }
zeromq-vi(){       vi $(zeromq-source) ; }
zeromq-env(){      elocal- ; }
zeromq-usage(){ cat << EOU

ZEROMQ
======


* http://zguide.zeromq.org/py:all


INSTALLS
-----------

D : as Chroma pyzmq dependency 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ZMQ that came with Chroma/pyzmq is 4.0.3 but where are the headers?::

    delta.local /usr/local/env/chroma_env/lib/python2.7/site-packages/zmq/__init__.pyc 4.0.3 14.0.1

Below headers refernce zmq.h but cannot find it::

    (chroma_env)delta:chroma_env blyth$ find $VIRTUAL_ENV -name zmq*.h
    /usr/local/env/chroma_env/lib/python2.7/site-packages/zmq/utils/zmq_compat.h
    /usr/local/env/chroma_env/lib/python2.7/site-packages/zmq/utils/zmq_constants.h




EOU
}
zeromq-dir(){ echo $(local-base)/env/zeromq/$(zeromq-name) ; }
zeromq-cd(){  cd $(zeromq-dir); }
zeromq-mate(){ mate $(zeromq-dir) ; }

zeromq-name(){ echo zeromq-4.0.4 ; }
zeromq-url(){  echo http://download.zeromq.org/$(zeromq-name).tar.gz ; }
zeromq-get(){
    local dir=$(dirname $(zeromq-dir)) &&  mkdir -p $dir && cd $dir
    local url=$(zeromq-url)
    local tgz=$(basename $url)
    local nam=${tgz/.tar.gz}
    [ ! -f "$tgz" ] && curl -O $url
    [ ! -d "$nam" ] && tar zxvf $nam
}


zeromq-make(){
  zeromq-cd
  ./configure --prefix=$VIRTUAL_ENV
  make 
  make install
}


zeromq-zguide-get(){
  git clone --depth=1 git://github.com/imatix/zguide.git
}

zeromq-versions(){
   python -c "import zmq, socket ; print socket.gethostname(), zmq.__file__, zmq.zmq_version(), zmq.pyzmq_version() "
}

