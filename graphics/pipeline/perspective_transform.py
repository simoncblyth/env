#!/usr/bin/env python
"""

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


class PerspectiveTransform(Transform):
    def __init__(self, view=None, near=0.1, far=100, yfov=30, nx=640, ny=480, flip=False ):
        """
        :param eye:   (x,y,z) position of camera/eye in world frame
        :param look:  (x,y,z) position of target in world frame
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
        if view is None:
            view = ViewTransform()
        self.setView( view )
        self.setCamera( near, far, yfov, nx, ny, flip )

    def setView(self, view):
        self.set('view',view)

    def setCamera(self, near, far, yfov, nx, ny, flip ):
        self.set('near',near)
        self.set('far',far)
        self.set('yfov',yfov)
        self.set('nx',nx)
        self.set('ny',ny)
        self.set('flip',flip)

    def copy(self):
        return PerspectiveTransform(self.view,self.near,self.far, self.yfov, self.nx, self.ny, self.flip)

    screensize = property(lambda self:(int(self.nx), int(self.ny)))
    aspect = property(lambda self:float(self.ny)/float(self.nx)) 
    top = property(lambda self:self.near * math.tan( self.yfov * 0.5 * math.pi / 180. ))
    right = property(lambda self:self.top*self.aspect) 

    def _calculate_matrix(self):
        #cw = world_to_camera(eye, look, up )
        cw = self.view.matrix

        oc = camera_to_orthographic( self.near, self.far )

        left = -self.right 
        bottom = -self.top
        co = orthographic_to_canonical( left, bottom, self.near,  self.right, self.top, self.far )

        sc = canonical_to_screen( float(self.nx), float(self.ny), self.flip  )

        sw = sc.dot(co).dot(oc).dot(cw)

        return sw
        #return world_to_screen_symmetric( self.eye, self.look, self.up, self.near, self.far, self.yfov, self.nx, self.ny, self.flip)

    def add(self, name, delta):
        init = getattr(self, name)
        #if name in ('eye','look',):
        #    self.set(name, init+np.array(delta))
        #else:
        self.set(name, init+delta)





if __name__ == '__main__':
    pt = PerspectiveTransform()
    print pt.matrix
    print pt.quaternion

    print pt( pt.look )




