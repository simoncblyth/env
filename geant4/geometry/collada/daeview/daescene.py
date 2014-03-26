#!/usr/bin/env python
"""
Coordination class


"""


from daetrackball import DAETrackball
from env.graphics.pipeline.unit_transform import UnitTransform, KeyView
#from env.graphics.pipeline.view_transform import ViewTransform


class DAEScene(object):
    def __init__(self, args, geometry ):

        self.args = args
        self.geometry = geometry  
        self.mesh = geometry.mesh
        self.view = None

        self.kwa = {}
        self.configure_target()
        self.configure_base()
        if not self.target is None:
            self.configure_lookat() 

        self.trackball = DAETrackball(**self.kwa)

    def configure_target(self):
        if self.args.target is None:
            target = None
        else:
            target = self.geometry.find_solid(self.args.target) 
            assert target, "failed to find target for argument %s " % self.args.target
        self.target = target 

    def configure_base(self):
        args = self.args
        self.kwa['thetaphi'] = args.thetaphi
        self.kwa['xyz'] = args.xyz
        self.kwa['yfov'] = args.yfov
        self.kwa['near'] = args.near
        self.kwa['far'] = args.far
        self.kwa['parallel'] = args.parallel

    def configure_lookat(self):
        """
        Convert eye/look/up input parameters into world coordinates
        """
        lower, upper, extent = self.target.bounds_extent
        unit = UnitTransform([lower,upper])

        self.view  = KeyView( self.args.eye, self.args.look, self.args.up, unit )
        eye, look, up = self.view._eye_look_up

        self.kwa['lookat'] = True
        self.kwa['extent'] = extent
        self.kwa['eye'] = eye
        self.kwa['look'] = look
        self.kwa['up'] = up

    def dump(self):
        if self.view:
            print "view\n", self.view
        if self.mesh: 
            print "full mesh\n",self.mesh.smry()
        if self.target: 
            print "target mesh\n",self.target.smry()


if __name__ == '__main__':
    pass


