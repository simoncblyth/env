#!/bin/bash -l

export-
export-export  

graphicstools-
graphicstools-export

chroma-

csa-
csa-export

zmq-
zmq-broker-export

zmqtunnelnode=$(zmq-tunnel-node-parse-cmdline "$*")

[ -n "$zmqtunnelnode" ] && zmq-tunnel-open ZMQ_BROKER_URL_BACKEND $zmqtunnelnode

g4daeview.py $*

[ -n "$zmqtunnelnode" ] && zmq-tunnel-close




