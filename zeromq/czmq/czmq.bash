# === func-gen- : zeromq/czmq/czmq fgp zeromq/czmq/czmq.bash fgn czmq fgh zeromq/czmq
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

Dependencies
-------------

* libzmq
* optionally : uuid-devel pkg


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
   czmq-icd
   zeromq-
   ./configure --with-libzmq=$(zeromq-prefix) --prefix=$(czmq-prefix)
}
czmq-make(){
   czmq-icd
   make  $*
}
czmq-install(){ czmq-make install ; }

czmq-build(){

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
czmq-broker-env(){ echo FRONTEND=tcp://*:$(zmq-frontend-port) BACKEND=tcp://*:$(zmq-backend-port) ; }

czmq-client(){ type $FUNCNAME ; FRONTEND=tcp://$(zmq-broker-host):$(zmq-frontend-port) $(czmq-bin czmq_client) ; }
czmq-worker(){ type $FUNCNAME ;  BACKEND=tcp://$(zmq-broker-host):$(zmq-backend-port)  $(czmq-bin czmq_worker) ; }

czmq-broker(){ 
   local cmd="$(czmq-broker-env) $(czmq-bin czmq_broker)"  
   echo $cmd
   eval $cmd
}


czmq-broker-env-sv(){ czmq-broker-env | tr " " "," ; }
czmq-broker-start-cmd(){
  cat << EOC
$(czmq-bin czmq_broker) 
EOC
}

czmq-broker-log(){ echo /tmp/env/zeromq/czmq/cmzq_broker.log ; }
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
  sv- 
  $FUNCNAME- | sv-plus czmq_broker.ini
}

