#!/usr/bin/env python

import os, logging
import numpy as np

from env.numerics.npy.nload import A
from env.numerics.npy.base import idp_

class CGDMLDetector(object):
    def __init__(self):
        self.gtransforms = np.load(idp_("CGDMLDetector/0/gtransforms.npy"))
        self.ltransforms = np.load(idp_("CGDMLDetector/0/ltransforms.npy"))

    def __repr__(self):
        return "\n".join([
              "gtransforms %s " % repr(self.gtransforms.shape),
              "ltransforms %s " % repr(self.ltransforms.shape)
              ])

    def getGlobalTransform(self, frame):
        return self.gtransforms[frame]

if __name__ == '__main__':

    frame = 3153

    det = CGDMLDetector()
    print det
    mat = det.getGlobalTransform(frame)
    print "mat %s " % repr(mat)


