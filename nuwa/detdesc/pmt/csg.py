#!/usr/bin/env python
import logging, hashlib, sys, os
import numpy as np
np.set_printoptions(precision=2) 

log = logging.getLogger(__name__)

class CSG(object):
    def __init__(self, elem, children=[]):
        """
        :param elem: 
        :param children: comprised of either base elements or other CSG nodes 
        """
        self.elem = elem
        self.children = children

    def __repr__(self):
        return "CSG node %s children [%s] " % ( len(self.children), repr(self.elem)) + "\n" + "\n".join(map(lambda _:"    " + repr(_), self.children))



 
