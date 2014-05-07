#!/usr/bin/env python
"""
Invoke with `echoserver.sh` for envvar setup.
"""
import os, time, logging
log = logging.getLogger(__name__)

import zmq  # invoke with echoserver.sh for env setup, python picking, ...
from env.zmqroot.serialize import serialize, deserialize
from env.chroma.ChromaPhotonList.cpl import examine_cpl, random_cpl

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(name)-20s:%(lineno)-3d %(message)s")
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    config = os.environ['ECHO_SERVER_CONFIG']
    socket.bind( config )

    log.info("bound to %s hwm %s " % ( config, socket.hwm ))

    while True:
        msg = socket.recv(copy=False)
        print repr(msg)
        log.info("recv message of length %s " % len(msg))

        obj = deserialize( msg.bytes )
        examine_cpl( obj )

        cpl = random_cpl()
        newmsg = serialize(cpl)

        time.sleep(1)
        log.info("now send")
        socket.send(newmsg)
    pass
pass

if __name__ == '__main__':
    main()
   




