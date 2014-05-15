#!/usr/bin/env python
"""
"""
import logging
log = logging.getLogger(__name__)

from env.chroma.chroma_propagator.propagator import Propagator

class DAEPropagator(object):
    def __init__(self, config, ctx ):
        self.config = config
        self.propagator = Propagator(ctx.gpu_geometry, config )

    def step(self, photons, max_steps=1):
        """
        :return: chroma.event.Photons instance
        """
        return self.propagator.propagate( photons, max_steps=max_steps )


if __name__ == '__main__':
    pass

