# === func-gen- : zeromq/zmq/zmq fgp zeromq/zmq/zmq.bash fgn zmq fgh zeromq/zmq
zmq-src(){      echo zeromq/zmq/zmq.bash ; }
zmq-source(){   echo ${BASH_SOURCE:-$(env-home)/$(zmq-src)} ; }
zmq-vi(){       vi $(zmq-source) ; }
zmq-env(){      elocal- ; }
zmq-usage(){ cat << EOU

ZMQ : Low Level C API 
=======================

This is for low level ZMQ C API usage/examples.

See also:

*zeromq-*
    getting/building/installing  etc.. 

*czmq-*
    higher level C binding

*pyzmq-*
    python binding


client/broker/worker topology
-----------------------------

Broker sits in the middle between client(s) and worker(s).
The client REQuest goes to the ROUTER socket and worker REPly returns to DEALER socket. 
The broker asynchronously monitors the ROUTER/DEALER sockets internally, 
and ensures that replys are sent back to the client that requested it.
ROUTER/DEALER sockets are non-blocking variants of the blocking REQ/REP sockets. 

NB unlike simple client/server REQ/REP between two nodes, the workers
need to **connect** (not **bind**) to their sockets. 

#. client **connects** to frontend, configured by envvar such as:

   * FRONTEND=tcp://ip.addr.of.broker:5001

#. worker **connects** to backend, configured by envvar:

   * BACKEND=tcp://ip.addr.of.broker:5002

#. broker running on node ip.addr.of.broker **binds** locally 
   to frontend and backend sockets, configured by envvars such as:

   * FRONTEND=tcp://*:5001  (the ROUTER socket)
   * BACKEND=tcp://*:5002   (the DEALER socket) 
   

The big advantage is that clients and workers need to know nothing 
about each other, meaning that clients and workers can come and go with no
need for reconfiguration. 

Only the "stable" broker IP address and relevant frontend/backend port 
needs be known by clients/workers.



EOU
}
zmq-dir(){ echo $(env-home)/zeromq/zmq ; }
zmq-bindir(){ echo $(local-base)/env/bin ; }

zmq-cd(){  cd $(zmq-dir); }
zmq-scd(){  cd $(zmq-dir); }
zmq-mate(){ mate $(zmq-dir) ; }

zmq-bin(){ echo $(zmq-bindir)/$1 ; }
zmq-cc(){
   local name=$1
   local bin=$(zmq-bin $name)
   mkdir -p $(dirname $bin)
   echo $msg compiling $bin 
   zeromq-
   cc $name.c -o $bin \
         -I$(zeromq-prefix)/include \
         -L$(zeromq-prefix)/lib -lzmq \
         -Wl,-rpath,$(zeromq-prefix)/lib
}

zmq-cc-build(){
  echo $msg building 
  zmq-scd
  local line
  ls -1 *.c | while read line ; do 
     local name=${line/.c}
     zmq-cc $name
  done
}


zmq-frontend-port(){ echo 5001 ; }
zmq-backend-port(){  echo 5002 ; }
zmq-broker-host(){ echo ${ZMQ_BROKER_HOST:-localhost} ; }

zmq-broker(){ FRONTEND=tcp://*:$(zmq-frontend-port) BACKEND=tcp://*:$(zmq-backend-port) $(zmq-bin zmq_broker) ; }
zmq-client(){ FRONTEND=tcp://$(zmq-broker-host):$(zmq-frontend-port) $(zmq-bin zmq_client) ; }
zmq-worker(){ BACKEND=tcp://$(zmq-broker-host):$(zmq-backend-port)  $(zmq-bin zmq_worker) ; }


