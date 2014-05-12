#!/bin/bash -l

zmqroot-
zmqroot-export

zmq-
zmq-broker-export
zmq-broker-info

cpl-
cpl-export

chroma-

python $(cpl-sdir)/cpl_responder.py


