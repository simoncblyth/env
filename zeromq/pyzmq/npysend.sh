#!/bin/bash -l

export-
export-export  

chroma-

zmq-
zmq-broker-export

zmqtunnelnode=$(zmq-tunnel-node-parse-cmdline "$*")

[ -n "$zmqtunnelnode" ] && zmq-tunnel-open ZMQ_BROKER_URL_FRONTEND $zmqtunnelnode

npysend.py $*

[ -n "$zmqtunnelnode" ] && zmq-tunnel-close



