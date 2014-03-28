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


class DAEScene(object):
    def __init__(self, geometry, config ):
        """

        near/far are in eye space, so they should 

        """
        args = config.args
        self.config = config
        self.geometry = geometry  

        #meshextent = geometry.mesh.extent # of all nodes loaded

        self.view = self.change_view( args.target )
        if args.jump:
            self.view = self.interpolate_view(args.jump)

        scaled_mode = args.target is None  
        xyz = args.xyz if scaled_mode else (0,0,0)

        self.scaled_mode = scaled_mode   

        self.trackball = DAETrackball( thetaphi=args.thetaphi, xyz=xyz )
        self.camera = DAECamera( size=args.size, near=args.near, far=args.far, yfov=args.yfov, nearclip=args.nearclip, farclip=args.farclip, yfovclip=args.yfovclip )

    def __repr__(self):
        return ""

    def bookmark(self):
        log.info("bookmark") 
        print self.view.current_view

    def external_message(self, msg ):
        pass
        log.info("external_message [%s]" % msg) 
        elems = msg.split(" ")
        if len(elems)==2:
            if elems[0] == "-j":
                view = self.interpolate_view(elems[1]) 
                self.view = view
                log.info("external_message triggered interpolated_view, press M to run the movie" )
            elif elems[0] == "-t":
                view = self.change_view(elems[1]) 
                log.info("external_message triggered change_view" )
                self.view = view
                #self.frame.redraw() 
            else:
                log.info("dont understand the message" )
        else:
            log.info("expecting two element msg, not [%s]" % msg )   


    def change_view(self, tspec):
        view = self.geometry.make_view( tspec, self.config.args )
        log.info("change_view tspec[%s]" % tspec  )
        return view
     
    def interpolate_view(self, jspec):
        views  = [self.view.current_view]
        views += [self.geometry.make_view( j, self.config.args ) for j in jspec.split(":")]
        view = DAEInterpolateView(views)
        log.info("interpolated_view movie sequence with %s views " % len(views))
        return view

    def dump(self):
        print "view\n", self.view
        print "trackball\n", self.trackball


if __name__ == '__main__':
    pass


