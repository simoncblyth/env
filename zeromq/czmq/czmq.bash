czmq-src(){      echo zeromq/czmq/czmq.bash ; }
czmq-source(){   echo ${BASH_SOURCE:-$(env-home)/$(czmq-src)} ; }
czmq-vi(){       vi $(czmq-source) ; }
czmq-usage(){ cat << EOU

CZMQ : High-level C Binding for 0MQ
=====================================

* http://czmq.zeromq.org
* http://czmq.zeromq.org/page:get-the-software
* https://github.com/zeromq/czmq

* https://github.com/zeromq/pyczmq


Installing broker onto a fresh node G5
-----------------------------------------

Get, build and install on G5::

    zeromq-
    zeromq--

    czmq-
    czmq--

    czmq-cc-build    # compile broker 

Start broker on G5 with local argument, as using SSH tunnelling is more robust::

    czmq_broker.sh local  


On D (GPU node) start the worker and tunnel to broker node::

    g4daechroma.sh --zmqtunnelnode=G5



Dependencies
-------------

* libzmq
* optionally : uuid-devel pkg


FUNCTIONS
---------

*czmq_worker_tunneled*
     opens ssh tunnel to the broker node and forwards a random available local port 
     to broker BACKEND on the broker node through the tunnel. Subsequently starts the 
     worker using the local addr rather than the direct remote one in order
     to route the traffic through the tunnel 
 
*czmq_client_tunneled*
     opens ssh tunnel to the broker node and forwards a random available local port 
     to broker FRONTEND on the broker node through the tunnel. Subsequently starts the 
     client using the local addr rather than the direct remote one in order
     to route the traffic through the tunnel 
 


czmq_broker supervisord hookup on N
-------------------------------------

#. *reread* then *update* to get the new process going  

::

    [blyth@belle7 env]$ czmq-
    [blyth@belle7 env]$ czmq-broker-sv 
    ...
    [blyth@belle7 env]$ sv
    N> reread
    czmq_broker: available
    N> start czmq_broker
    czmq_broker: ERROR (no such process)
    N> help reload
    reload      Restart the remote supervisord.
    N> help update
    update      Reload config and add/remove as necessary
    N> update
    czmq_broker: added process group
    N> status           
    czmq_broker                      RUNNING    pid 8651, uptime 0:00:40
    daeserver                        RUNNING    pid 10122, uptime 21 days, 3:12:51
    ...
    nginx                            RUNNING    pid 2285, uptime 86 days, 0:23:32
    N> 
    N> tail -f czmq_broker
    ==> Press Ctrl-C to exit <==
    14-05-14 14:55:40 I: /data1/env/local/env/bin/czmq_broker starting 
    14-05-14 14:55:40 I: binding frontend ROUTER:[tcp://*:5001]
    14-05-14 14:55:40 I: binding backend DEALER:[tcp://*:5002]


EOU
}
czmq-env(){    
    elocal- 
    zmq-   # for client/broker/worker config
}
czmq-fold(){ echo $(local-base)/env/zeromq/czmq ; }
czmq-dir(){ echo $(czmq-fold)/$(czmq-name) ; }
czmq-sdir(){ echo $(env-home)/zeromq/czmq ; }

czmq-bindir(){ echo $(local-base)/env/bin ; }

czmq-cd(){  cd $(czmq-sdir); }
czmq-icd(){  cd $(czmq-dir); }
czmq-scd(){  cd $(czmq-sdir); }

#czmq-version(){ echo 4.2.0 ; }   ## circa 2020
czmq-version(){ echo 2.0.3 ; }   ## circa 2015

czmq-name(){  echo czmq-$(czmq-version) ; }

czmq-url(){
   case $(czmq-version) in 
      2.*) czmq-archive-url ;;
      4.*) czmq-release-url ;;
   esac
}
czmq-release-url(){ echo https://github.com/zeromq/czmq/releases/download/v$(czmq-version)/czmq-$(czmq-version).tar.gz ; }
czmq-archive-url(){ echo https://archive.org/download/zeromq_czmq_$(czmq-version)/czmq-$(czmq-version).tar.gz ; }


czmq-info(){  cat << EOI

    czmq-version  : $(czmq-version)
    czmq-url      : $(czmq-url)
    czmq-dir      : $(czmq-dir)

EOI
}

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
   czmq-icd
   zeromq-
   ./configure --with-libzmq=$(zeromq-prefix) --prefix=$(czmq-prefix)
}
czmq-make(){
   czmq-icd
   make  $*
}
czmq-install(){ czmq-make install ; }

czmq--(){
   czmq-get
   czmq-configure
   czmq-make
   czmq-install
}


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
         -Wl,-rpath,$(czmq-prefix)/lib \
         -Wl,-rpath,$(zeromq-prefix)/lib
}


czmq-cc-build(){
  echo $msg building examples
  czmq-scd
  local line
  ls -1 *.c | while read line ; do 
     local name=${line/.c}
     czmq-cc $name
  done

}

# NB using config from zmq- for interopability 


czmq-info-old()
{
  local names="czmq-broker-env czmq-client-addr czmq-worker-addr"
  for name in $names ; do
     echo $name $($name)
  done 
}

czmq-broker-env(){ echo FRONTEND=tcp://*:$(zmq-frontend-port) BACKEND=tcp://*:$(zmq-backend-port) ; }
czmq-client-addr(){ echo $(zmq-broker-host):$(zmq-frontend-port) ; }
czmq-worker-addr(){ echo $(zmq-broker-host):$(zmq-backend-port) ; }

czmq-sshsrv(){ echo N ; }


czmq-broker-info(){ cat << EOI



   czmq-client-  $(czmq-client-)
       client creates ZMQ_REQ socket, sends request and waits for reply 

$(cat -n $(czmq-sdir)/czmq_client.c | tail -30)


   czmq-worker-  $(czmq-worker-)

$(cat -n $(czmq-sdir)/czmq_worker.c | tail -30)



   czmq-broker-local-  $(czmq-broker-local-)

   czmq-broker-  $(czmq-broker-)
       intermediates between clients and workers
       ROUTER/DEALER allows REQ/REP to go across the proxy 

$(cat -n $(czmq-sdir)/czmq_broker.c | tail -30)


EOI
}

czmq-client-(){ cat << EOC
FRONTEND=tcp://$(czmq-client-addr) $(czmq-bin czmq_client)
EOC
}
czmq-worker-(){ cat << EOC
BACKEND=tcp://$(czmq-worker-addr) $(czmq-bin czmq_worker)
EOC
}
czmq-broker-(){  cat << EOC
$(czmq-broker-env) $(czmq-bin czmq_broker)
EOC
}
czmq-broker-local-(){ cat << EOC
ZMQ_BROKER_TAG=SELF $(czmq-broker-env) $(czmq-bin czmq_broker)
EOC
}

czmq-client(){ 
   local cmd=$($FUNCNAME-)
   echo $cmd 
   eval $cmd
}
czmq-worker(){
   local cmd=$($FUNCNAME-)
   echo $cmd 
   eval $cmd
}
czmq-broker(){ 
   local cmd="$($FUNCNAME-)"  
   echo $cmd
   eval $cmd
}
czmq-broker-local(){
   local cmd="$($FUNCNAME-)"  
   echo $cmd
   eval $cmd
}



czmq-tunnel-cmd(){
   local laddr=$1
   local raddr=$2
   local tcmd="ssh -fN -p 22 -L ${laddr}:${raddr} "
   echo $tcmd
}

czmq-tcp(){
    lsof | grep TCP
}

czmq-worker-tunneled(){
   local raddr=$(czmq-worker-addr)
   local laddr="127.0.0.1:$(available_port.py)" 

   local tcmd="$(czmq-tunnel-cmd $laddr $raddr) $(czmq-sshsrv)"
   echo $tcmd 
   eval $tcmd

   local cmd="BACKEND=tcp://${laddr} $(czmq-bin czmq_worker)"
   echo $cmd 
   eval $cmd
}

czmq-client-tunneled(){
   local raddr=$(czmq-client-addr)
   local laddr="127.0.0.1:$(available_port.py)" 

   local tcmd="$(czmq-tunnel-cmd $laddr $raddr) $(czmq-sshsrv)"
   echo $tcmd 
   eval $tcmd

   local cmd="FRONTEND=tcp://${laddr} $(czmq-bin czmq_client)"
   echo $cmd 
   eval $cmd
}


czmq-main(){
    local arg=${1:-local}
    zmq-broker-info
    if [ "$arg" == "local" ]; then
        type czmq-broker-local
        czmq-broker-local
    else
        type czmq-broker
        czmq-broker
    fi
}




czmq-broker-env-sv(){ czmq-broker-env | tr " " "," ; }
czmq-broker-start-cmd(){
  cat << EOC
$(czmq-bin czmq_broker) 
EOC
}

#czmq-broker-log(){ echo /tmp/env/zeromq/czmq/cmzq_broker.log ; }
czmq-broker-log(){ echo $(local-base)/env/zeromq/czmq/cmzq_broker.log ; }
czmq-broker-tail(){ tail -f $(czmq-broker-log) ; }
czmq-broker-sv-(){ 

mkdir -p $(dirname $(czmq-broker-log))
cat << EOX
[program:czmq_broker]
environment=$(czmq-broker-env-sv)
command=$(czmq-broker-start-cmd)
process_name=%(program_name)s
autostart=true
autorestart=true

redirect_stderr=true
stdout_logfile=$(czmq-broker-log)
stdout_logfile_maxbytes=5MB
stdout_logfile_backups=10

EOX
}
czmq-broker-sv(){
  local msg="=== $FUNCNAME :"
  local log=$(czmq-broker-log)
  local dir=$(dirname $log)
  [ ! -d "$dir" ] && echo $msg creating $dir && mkdir -p $dir 

  sv- 
  $FUNCNAME- | sv-plus czmq_broker.ini
}

