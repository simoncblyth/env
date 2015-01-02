#!/usr/bin/env python
"""

"""
import logging, pprint
log = logging.getLogger(__name__)

import OpenGL.GL as gl

import numpy as np
from daephotonsrenderer import DAEPhotonsRenderer
from daephotonsstyler import DAEPhotonsStyler


from daedrawable import DAEDrawable
from env.g4dae.types import VBOStep


class DAEGenstepAnalyzer(object):
    def __init__(self, drawable):
        self.drawable = drawable
     
    def _get_counts_firsts_drawcount(self):
        """
        Counts with truncation, indices of start of each photon record
        """
        nitem = len(self.drawable.array)
        counts = np.tile(1, nitem).astype(np.int32)
        firsts = np.arange(nitem, dtype='i')*self.drawable.max_slots
        return counts, firsts, nitem
    counts_firsts_drawcount = property(_get_counts_firsts_drawcount, doc=_get_counts_firsts_drawcount.__doc__)


class DAEGenstep(DAEDrawable):
    """
    """
    animate = False
    def __init__(self, _array, event ):
        """
        :param genstep: 
        :param event: `DAEEventBase` instance
        """ 
        DAEDrawable.__init__(self, _array, event ) 
        self.style = "noodles"

        self.numquad = VBOStep.numquad  # fundamental nature of VBO data structure, not a "parameter"
        self.max_slots = 1
        assert self.config.args.numquad == self.numquad
        self.analyzer = DAEGenstepAnalyzer(self)

    def handle_array(self, _array):
        return VBOStep.vbo_from_array(_array, self.max_slots)    


 
