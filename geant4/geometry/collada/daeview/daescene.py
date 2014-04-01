#!/usr/bin/env python
"""
Coordination class

Seek to split trackball into separate rotation
and translation portions.


Next
-----

#. CUDA/Chroma OpenGL interop, experiment with PBOs

#. light positioning using appropriate top level mesh transforms

#. placemarks, write the commandline to return to a viewpoint and camera configuration
   in response to a key press, this will entail world2model transforms to get the 
   parameters in model frame of the current target volume

#. visual bounding boxes

#. trackball home, to return to current target standard position 

   * trackball translations and rotations apply on top of the view lookAt transformation 
     so when animating between views this means can be offset from the view sequence.  

#. animation speed control, speed dependant on distance 

#. solid picking, click and see material/surface properties 

   * gluUnProject gives world space coordinates from mouse position and the matrices
   * find the deepest volume bbox that contains the clicked point 
   * interactive target switching 

"""
import logging
log = logging.getLogger(__name__)
import numpy as np
from daetrackball import DAETrackball
from daecamera import DAECamera
from daeinterpolateview import DAEInterpolateView
from daeviewpoint import DAEViewpoint
from daeutil import Transform
from daelights import DAELights



ivec_ = lambda _:map(int,_.split(","))
fvec_ = lambda _:map(float,_.split(","))

class DAEScene(object):
    def __init__(self, geometry, config ):
        self.geometry = geometry  
        self.config = config
        args = config.args
        #
        self.scaled_mode = args.target is None  
        xyz = args.xyz if self.scaled_mode else (0,0,0)

        self.trackball = DAETrackball( thetaphi=fvec_(args.thetaphi), xyz=xyz, trackballradius=args.trackballradius, translatefactor=args.translatefactor )

        self.view = self.change_view( args.target , prior=None)
        if args.jump:
            self.view = self.interpolate_view(args.jump)

        camera = DAECamera( size=ivec_(args.size), 
                            near=args.near, 
                            far=args.far, 
                            yfov=args.yfov, 
                            nearclip=fvec_(args.nearclip), 
                            farclip=fvec_(args.farclip), 
                            yfovclip=fvec_(args.yfovclip) )

        self.camera = camera 
        print self.view.smry()
        
        if not self.scaled_mode:
            kscale = config.args.kscale
            light_transform = geometry.mesh.model2world 
        else:
            kscale = 1.
            light_transform = Transform()

        self.kscale = kscale
        self.lights = DAELights( light_transform, config )
        self.solids = []

    def where(self):
        """
        """ 
        model_xyz = self.view.offset_eye_position( self.trackball.xyz ) 
        return model_xyz

    def __repr__(self):
        w = self.where()
        return "SC %5.1f %5.1f %5.1f " % (w[0],w[1],w[2])

    def clicked_point(self, click ):
        """
        :param click: world frame xyz 

        Find solids that contain the click coordinates,  
        sorted by extent.
        """ 
        #log.info("clicked point %s " % repr(click) ) 
        indices = self.geometry.find_bbox_solid( click )
        solids = sorted([self.geometry.solids[_] for _ in indices],key=lambda _:_.extent) 
        #print "\n".join(map(repr, solids))
        self.solids = solids

    def bookmark(self):
        log.info("bookmark") 
        print self.view.current_view

    def external_message(self, msg ):
        live_args = self.config( msg )
        if live_args is None:
            log.warn("external_message [%s] PARSE ERROR : IGNORING " % str(msg)) 
            return
        pass
        log.info("external_message [%s] [%s]" % (msg,str(live_args))) 

        newview = None
        elu = {}
        for k,v in vars(live_args).items():
            if k == "target":
                newview = self.change_view(v, prior=self.view ) 
            elif k == "jump":
                newview = self.interpolate_view(v) 
            elif k == "ajump":
                newview = self.interpolate_view(v, append=True) 
            elif k in ("eye","look","up") :
                elu[k] = v
            elif k == "kscale":
                self.kscale = kscale
            elif k in ("near","far","yfov","nearclip","farclip","yfovclip"):
                setattr(self.camera, k, v )
            elif k in ("translatefactor","trackballradius"):
                setattr(self.trackball, k, v )
            else:
                log.info("handling of external message key [%s] value [%s] not yet implemented " % (k,v) )
            pass
        pass

        if newview is None:
            log.debug("view unchanged by external message")
        else:
            log.info("view changed by external message")
            self.view = newview

        if len(elu) > 0:
            log.info("changing parameters of existing view %s " % repr(elu)) 
            self.view.current_view.change_eye_look_up( **elu )
 

    def change_view(self, tspec, prior=None):
        log.info("change_view tspec[%s]" % tspec  )
        self.trackball.home()
        return DAEViewpoint.make_view( self.geometry, tspec, self.config.args, prior=prior )
     
    def interpolate_view(self, jspec, append=False):
        self.trackball.home()
        views  = self.view.views if append else [self.view.current_view]
        views += [DAEViewpoint.make_view( self.geometry, j, self.config.args, prior=views[-1] ) for j in jspec.split(":")]
        log.info("interpolated_view append %s movie sequence with %s views " % (append,len(views)))

        interpolateview = None
        if len(filter(None, views)) != len(views):
            log.warn("interpolate_view encountered None view : geometry nodes %s not loaded ? " % jspec ) 
        else:
            interpolateview = DAEInterpolateView(views)
        pass
        return interpolateview

    def dump(self):
        print "view\n", self.view
        print "trackball\n", self.trackball


if __name__ == '__main__':
    pass


