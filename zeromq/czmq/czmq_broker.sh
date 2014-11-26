#!/bin/bash -l

zmq-
zmq-broker-info

czmq-


if [ "$1" == "local" ]; then 
   type czmq-broker-local
   czmq-broker-local
else
   type czmq-broker
   czmq-broker
fi



