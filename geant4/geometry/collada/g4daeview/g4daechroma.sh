#!/bin/bash -l
#echo $0 $(date)


cmdline="$*"
zmqtunnelnode=""
cudagdb=0

if [ "${cmdline/--zmqtunnelnode}" != "${cmdline}" ]; then
   for arg in $cmdline ; do
       case $arg in
          --zmqtunnelnode=*)  zmqtunnelnode=${1#*=} ;; 
       esac      
   done
fi 

if [ "${cmdline/--cuda-gdb}" != "${cmdline}" ]; then
   cudagdb=1
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


if [ "$NODE_TAG" == "N" ]; then
   #
   # for parasitic use of NuWa python2.7 on N with easy_installed pyzmq, see pyzmq-
   # although no GPU/CUDA/Chroma on N, it would be instructive to have this
   # operational there to some extent to check CPL transport 
   # 
   fenv 
   export ZMQROOT_LIB=$DYB/NuWa-trunk/dybgaudi/InstallArea/$CMTCONFIG/lib/libZMQRoot.so
   export CHROMAPHOTONLIST_LIB=$DYB/NuWa-trunk/dybgaudi/InstallArea/$CMTCONFIG/lib/libChroma.so
   env | grep ZMQ 
   env | grep CHROMA
fi



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
#echo $0 $(date)

if [ "$cudagdb" == "1" ]; then
   cd $ENV_HOME/geant4/geometry/collada/g4daeview 
   cuda-gdb --args python -m pycuda.debug g4daechroma.py $*
else
   g4daechroma.py $*
fi


#cuda_info.py



