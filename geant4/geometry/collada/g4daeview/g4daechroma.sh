#!/bin/bash -l

cmdline="$*"
cudagdb=0

if [ "${cmdline/--cuda-gdb}" != "${cmdline}" ]; then
   cudagdb=1
fi 

export-
export-export  

chroma-

zmq-
zmq-broker-export

zmqtunnelnode=$(zmq-tunnel-node-parse-cmdline "$cmdline")

[ -n "$zmqtunnelnode" ] && zmq-tunnel-open ZMQ_BROKER_URL_BACKEND $zmqtunnelnode

csa-
csa-export


if [ "$cudagdb" == "1" ]; then
   cd $ENV_HOME/geant4/geometry/collada/g4daeview 
   cuda-gdb --args python -m pycuda.debug g4daechroma.py $*
else
   g4daechroma.py $*
fi

[ -n "$zmqtunnelnode" ] && zmq-tunnel-close


