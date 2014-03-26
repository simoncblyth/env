#!/usr/bin/env python
"""
Coordination class

Seek to split trackball into separate rotation
and translation portions.

"""
import logging
log = logging.getLogger(__name__)
import numpy as np
from daetrackball import DAETrackball
from daeinterpolateview import DAEInterpolateView


class DAEScene(object):
    def __init__(self, geometry, config ):
        """
        Intend to move to lookat/target mode all the time, 
        the scaled_mode flag is a crutch to allow use of the
        old scaled VBO approach 
        while target mode still not fully working. 
        """
        args = config.args
        self.config = config
        self.geometry = geometry  

        view = geometry.make_view( args.target, args.eye, args.look, args.up )
        self.extent = view.extent

        if args.jump:
            views  = [view]
            views += [geometry.make_view( jump, args.eye, args.look, args.up ) for jump in args.jump.split(",")]
            view = DAEInterpolateView(views)

        self.view = view        

        kwa = {}
        kwa['thetaphi'] = args.thetaphi
        kwa['yfov'] = args.yfov
        kwa['near'] = args.near
        kwa['far'] = args.far
        kwa['nearclip'] = args.nearclip
        kwa['farclip'] = args.farclip

        scaled_mode = args.target is None  
        kwa['xyz'] = args.xyz if scaled_mode else (0,0,0)

        self.scaled_mode = scaled_mode   # a crutch to be removed
        self.trackball = DAETrackball(**kwa)

    def __repr__(self):
        return ""

    def dump(self):
        print "view\n", self.view
        print "trackball\n", self.trackball





if __name__ == '__main__':
    pass


