#!/usr/bin/env python
"""
For usage example see `env/chroma/echoserver/echoserver.py`

"""
import os, logging
log = logging.getLogger(__name__)

from env.zmqroot.responder import ZMQRootResponder
from cpl import examine_cpl, random_cpl


class CPLResponder(ZMQRootResponder):
    def __init__(self, config ):
        ZMQRootResponder.__init__(self, config)

    def reply(self, obj):
        log.info("reply") 

        if self.config.dump:
            examine_cpl( obj )

        if self.config.random:
            obj = random_cpl()

        return obj 



def check_responder():
    """
    """
    class Config(object):
        mode = 'connect'
        endpoint = os.environ['ZMQ_BROKER_URL_BACKEND']
        filename = '/tmp/lastmsg.zmq'
        timeout = None # milliseconds  (None means dont timeout, just block)
        sleep = 0.5  # seconds
        dump = True
        random = False
    pass
    cfg = Config()
    responder = CPLResponder(cfg)
    log.info("polling: %s " % repr(responder))
    responder.poll()

def main():
    logging.basicConfig(level=logging.INFO)
    check_responder()

if __name__ == '__main__':
    main()



