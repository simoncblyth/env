# === func-gen- : zeromq/pyzmq/pyzmq fgp zeromq/pyzmq/pyzmq.bash fgn pyzmq fgh zeromq/pyzmq
pyzmq-src(){      echo zeromq/pyzmq/pyzmq.bash ; }
pyzmq-source(){   echo ${BASH_SOURCE:-$(env-home)/$(pyzmq-src)} ; }
pyzmq-vi(){       vi $(pyzmq-source) ; }
pyzmq-usage(){ cat << EOU

PYZMQ
=======

* PyZMQ works with Python 3 (>= 3.2), and Python 2 (>= 2.6)

* http://zeromq.org/bindings:python
* http://zeromq.github.io/pyzmq/


fix memory leak in zero-copy allocation 
https://github.com/zeromq/pyzmq/pull/517



EOU
}
pyzmq-dir(){ echo $(env-home)/zeromq/pyzmq ; }
pyzmq-cd(){  cd $(pyzmq-dir); }
pyzmq-mate(){ mate $(pyzmq-dir) ; }

pyzmq-env(){    
    elocal- 
    chroma-   # for the right python
    zmq-  # client/broker/worker config
}

pyzmq-operator(){ python $(pyzmq-dir)/zmq_operator.py $* ; }

pyzmq-broker(){ FRONTEND=tcp://*:$(zmq-frontend-port) BACKEND=tcp://*:$(zmq-backend-port) pyzmq-operator broker ; }
pyzmq-client(){ FRONTEND=tcp://$(zmq-broker-host):$(zmq-frontend-port) pyzmq-operator client ; }
pyzmq-worker(){ BACKEND=tcp://$(zmq-broker-host):$(zmq-backend-port)   pyzmq-operator worker ; }

