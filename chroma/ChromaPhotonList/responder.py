#!/usr/bin/env python
"""
For usage example see `env/chroma/echoserver/echoserver.py`

"""
import logging
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


if __name__ == '__main__':
    pass

