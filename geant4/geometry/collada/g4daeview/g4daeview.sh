#!/bin/bash -l

cmdline="$*"
zmqtunnelnode=""

if [ "${cmdline/--zmqtunnelnode}" != "${cmdline}" ]; then
   for arg in $cmdline ; do
       case $arg in
          --zmqtunnelnode=*)  zmqtunnelnode=${1#*=} ;; 
       esac      
   done
fi 

#echo zmqtunnelnode=\"$zmqtunnelnode\"

export-
export-export   # DAE_NAME_ envvar

chroma-

#cuda_info.py

zmqroot-
zmqroot-export 

cpl-
cpl-export

zmq-
zmq-broker-export


ssh-tunnel-cmd(){
   local laddr=$1
   local raddr=$2
   local tcmd="ssh -fN -p 22 -L ${laddr}:${raddr} "
   echo $tcmd
}

ssh-tunnel-open(){
  local msg="=== $FUNCNAME :" 
  local ssh_node=$1

  [ -z "${ZMQ_BROKER_URL_BACKEND}" ] && echo $msg WARNING : opening an ssh tunnel required envvar ZMQ_BROKER_URL_BACKEND && return 0 

  local url=${ZMQ_BROKER_URL_BACKEND}
  local raddr=${url/tcp:\/\/}
  local laddr="127.0.0.1:$(available_port.py)" 

  local tcmd="$(ssh-tunnel-cmd $laddr $raddr) ${ssh_node}"
  echo $msg opening ssh tunnel using below command
  echo $tcmd 
  eval $tcmd

  local lurl=tcp://$laddr 
  echo $msg modifying ZMQ_BROKER_URL_BACKEND to $lurl in order to use the ssh tunnel  
  export ZMQ_BROKER_URL_BACKEND=$lurl

}

[ -n "$zmqtunnelnode" ] && ssh-tunnel-open $zmqtunnelnode

#echo starting
g4daeview.py $*

#cuda_info.py


