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

TODO
-----

#. add monitoring to zmq_broker.c so message activity is logged, 
   using the third *capture* argument of the *zmq_proxy* call

#. configure zmq_broker to run continuously beneath supervisord on N 


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

dodgy outgoing network workaround
-----------------------------------

belle7 refuses to operate as client or worker talking to remote broker on ports 5001, 5002

* it seems that outgoing connections on "unusual" ports are blocked for belle7
* workaround is to open a tcp connection from remote client and get belle7 to 
  reply within that open connection  

Topology that works:

#. put broker on belle7::

   [blyth@belle7 ~]$ zmq-;zmq-broker
 
#. worker and client elsewhere (or on belle7 too):: 

   [blyth@cms02 ~]$ czmq-; ZMQ_BROKER_HOST=$(local-tag2ip N) czmq-worker
   delta:~ blyth$ zmq-; ZMQ_BROKER_HOST=$(local-tag2ip N) zmq-client


Connecting nuwa.py/Geant4 and g4daeview.py/Chroma
---------------------------------------------------

Configure nuwa.py/Geant4 as client sending REQ with ChromaPhotonList 
objects to the broker with *CSA_CLIENT_CONFIG* envvar::

     68 csa-nuwarun(){
     69 
     70    zmq-
     71    export CSA_CLIENT_CONFIG=$(zmq-broker-url)   
     72    nuwa.py -n 1 -m "fmcpmuon --chroma"
     73 
     74 }

Configure g4daeview.py/Chroma as worker receiving REP





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


zmq-broker-info(){  cat << EOI

   zmq-broker-url          : $(zmq-broker-url)
   zmq-broker-url-frontend : $(zmq-broker-url-frontend)
   zmq-broker-url-backend  : $(zmq-broker-url-backend)

   zmq-broker-host         : $(zmq-broker-host)

EOI
}
zmq-frontend-port(){ echo 5001 ; }
zmq-backend-port(){  echo 5002 ; }
zmq-broker-tag(){ echo ${ZMQ_BROKER_TAG:-SELF} ; }
zmq-broker-host(){ local-tag2ip $(zmq-broker-tag) ; }

zmq-broker-url(){ zmq-broker-url-frontend ; }
zmq-broker-url-frontend(){ echo tcp://$(zmq-broker-host):$(zmq-frontend-port) ;}
zmq-broker-url-backend(){  echo tcp://$(zmq-broker-host):$(zmq-backend-port) ;}

zmq-broker-export(){
   export ZMQ_BROKER_URL_FRONTEND=$(zmq-broker-url-frontend)
   export ZMQ_BROKER_URL_BACKEND=$(zmq-broker-url-backend)
}

zmq-broker(){ FRONTEND=tcp://*:$(zmq-frontend-port) BACKEND=tcp://*:$(zmq-backend-port) $(zmq-bin zmq_broker) ; }
zmq-client(){ FRONTEND=$(zmq-broker-url-frontend) $(zmq-bin zmq_client) ; }
zmq-worker(){  BACKEND=$(zmq-broker-url-backend)  $(zmq-bin zmq_worker) ; }


