# === func-gen- : zeromq/pyzmq/pyzmq fgp zeromq/pyzmq/pyzmq.bash fgn pyzmq fgh zeromq/pyzmq
pyzmq-src(){      echo zeromq/pyzmq/pyzmq.bash ; }
pyzmq-source(){   echo ${BASH_SOURCE:-$(env-home)/$(pyzmq-src)} ; }
pyzmq-vi(){       vi $(pyzmq-source) ; }
pyzmq-env(){      elocal- ; }
pyzmq-usage(){ cat << EOU

PYZMQ
=======

* PyZMQ works with Python 3 (>= 3.2), and Python 2 (>= 2.6)

* http://zeromq.org/bindings:python
* http://zeromq.github.io/pyzmq/


EOU
}
pyzmq-dir(){ echo $(local-base)/env/zeromq/pyzmq/zeromq/pyzmq-pyzmq ; }
pyzmq-cd(){  cd $(pyzmq-dir); }
pyzmq-mate(){ mate $(pyzmq-dir) ; }
pyzmq-get(){
   local dir=$(dirname $(pyzmq-dir)) &&  mkdir -p $dir && cd $dir

}
