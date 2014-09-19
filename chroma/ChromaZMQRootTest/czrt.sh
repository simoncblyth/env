#!/bin/bash -l


usage(){ cat << EOU

CZRT.SH
=========

::

   czrt.sh --zmqtunnelnode=N


EOU
}

cmdline="$*"
zmqtunnelnode=""

if [ "${cmdline/--zmqtunnelnode}" != "${cmdline}" ]; then
   for arg in $cmdline ; do
       case $arg in
          --zmqtunnelnode=*)  zmqtunnelnode=${1#*=} ;;  
       esac    
   done
fi 


czrt-

zmq-
zmq-broker-export
zmq-broker-info


ssh-tunnel-cmd(){
   local laddr=$1
   local raddr=$2
   local tcmd="ssh -fN -p 22 -L ${laddr}:${raddr} "
   echo $tcmd
}

ssh-tunnel-open(){
  local msg="=== $FUNCNAME :" 
  local ssh_node=$1

  [ -z "${ZMQ_BROKER_URL_FRONTEND}" ] && echo $msg WARNING : opening an ssh tunnel required envvar ZMQ_BROKER_URL_FRONTEND && return 0 

  local url=${ZMQ_BROKER_URL_FRONTEND}
  local raddr=${url/tcp:\/\/}
  local laddr="127.0.0.1:$(available_port.py)" 

  local tcmd="$(ssh-tunnel-cmd $laddr $raddr) ${ssh_node}"
  echo $msg opening ssh tunnel using below command
  echo $tcmd 
  eval $tcmd

  local lurl=tcp://$laddr
  echo $msg modifying ZMQ_BROKER_URL_FRONTEND to $lurl in order to use the ssh tunnel  
  export ZMQ_BROKER_URL_FRONTEND=$lurl

}


[ -n "$zmqtunnelnode" ] && ssh-tunnel-open $zmqtunnelnode



CHROMA_CLIENT_CONFIG=${ZMQ_BROKER_URL_FRONTEND} $(czrt-bin)

