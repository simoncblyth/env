#!/bin/bash -l

[ "$NODE_TAG" == "N" ] && fenv 




zmqroot-
zmqroot-export

zmq-
zmq-broker-export

chroma-


if [ "$NODE_TAG" == "N" ]; then
   export ZMQROOT_LIB=/data1/env/local/dyb/NuWa-trunk/dybgaudi/InstallArea/i686-slc5-gcc41-dbg/lib/libZMQRoot.so
fi

env | grep ZMQ
python ./responder.py


