#!/usr/bin/env python
"""
Perspective Transformation
==========================

Using the approach described by Wolfgang Hurst, especially 
in "GRAPHICS PIPELINE I: PERSPECTIVE PROJECTION"  Lecture 7
of the course. 

* http://www.cs.uu.nl/docs/vakken/gr/2012-13/gr_lectures.html
* http://www.cs.uu.nl/docs/vakken/gr/2012-13/Slides/INFOGR_2012-2013_lecture-07_projection_annotated.pdf

Videos linked from the lectures page.

Summary
--------

#. world_to_camera

   * camera frame positions eye at origin and look along -Z with Y being camera up, and X to camera right 

#. camera_to_orthographic

   * orthographic frame is an axis aligned box with extremities (l,b,n) (t,r,f)   

#. orthographic_to_canonical

   * canonical frame is axis aligned box with extremities (-1,-1,-1) (1,1,1)

#. canonical_to_screen

   * screen frame is of pixel dimensions from (0,0,z) (width, height,z) where z gets ignored


Other Refs
-----------

* http://en.wikipedia.org/wiki/Orthographic_projection_(geometry)


"""

import logging
log = logging.getLogger(__name__)

import numpy as np
import math
from transform import Transform

from world_to_screen import world_to_screen_symmetric


from world_to_camera import world_to_camera
from camera_to_orthographic import camera_to_orthographic
from orthographic_to_canonical import orthographic_to_canonical
from canonical_to_screen import canonical_to_screen

from view_transform import ViewTransform 
from unit_transform import UnitTransform 


class PerspectiveTransform(Transform):
    def __init__(self, view=None, near=0.1, far=100, yfov=30, nx=640, ny=480, flip=False, orthographic=0 ):
        """
        :param view:  ViewTransform instance
        :param near:  distance from eye to view frustrum near plane, in world frame dimensions 
                      (should be less than the distance to geometry of interest, but greater than zero, **never zero**)
        :param far:   distance from eye to frustrum far place, in world frame dimensions    
                      (should be somewhat greater than the distance to geometry of interest, but not huge)
        :param yfov:  camera vertical field of view in degress, eg 30, 45, 90
        :param nx:    left-right pixel dimensions
        :param ny:    bottom-top pixel dimensions
        :param flip:  when True makes top left correpond to pixel coordinate (0,0), instead of bottom left
        """
        Transform.__init__(self)
        self.setView( view )
        self.setCamera( near, far, yfov, nx, ny, flip, orthographic )

    def setView(self, view):
        self.set('view',view)

    def setCamera(self, near, far, yfov, nx, ny, flip, orthographic ):
        self.set('near',near)
        self.set('far',far)
        self.set('yfov',yfov)
        self.set('nx',nx)
        self.set('ny',ny)
        self.set('flip',flip)
        self.set('orthographic',orthographic )

    def copy(self):
        return PerspectiveTransform(self.view, self.near,self.far, self.yfov, self.nx, self.ny, self.flip, self.orthographic )

    screensize = property(lambda self:(int(self.nx), int(self.ny)))

    # aspect is the ratio of the width to the height of an image 
    aspect = property(lambda self:float(self.nx)/float(self.ny)) 
    top = property(lambda self:self.near * math.tan( self.yfov * 0.5 * math.pi / 180. ))
    right = property(lambda self:self.top*self.aspect) 

    def _calculate_matrix(self):
        """
        """
        cw = self.view.matrix    # eye at origin looking along -Z

        if self.orthographic > 0:
            # TODO base this scale somehow on near, rather than something extra injected in 
            oc = np.identity(4)
            ortho_scale = self.orthographic
            oc[0,0] = ortho_scale  
            oc[1,1] = ortho_scale
            oc[2,2] = 0 
        else:
            oc = camera_to_orthographic( self.near, self.far )   # perspective divide happens here


        # fov and aspect dependency comes in with the top, right, left, bottom 
        # hmm, kinda funny that fov still effects orthographic ? 
        left, bottom, near = -self.right, -self.top, self.near
        right,top,far  = self.right, self.top, self.far

        co = orthographic_to_canonical( left, bottom, near, right, top, far, debug=False )

        sc = canonical_to_screen( float(self.nx), float(self.ny), self.flip  )

        sw = sc.dot(co).dot(oc).dot(cw)

        return sw

    def add(self, name, delta):
        init = getattr(self, name)
        self.set(name, init+delta)



if __name__ == '__main__':
    pt = PerspectiveTransform()
    print pt.matrix
    print pt.quaternion

    print pt( pt.look )




