#!/usr/bin/env python


import logging
log = logging.getLogger(__name__)

import numpy as np
import math
from transform import Transform

norm_ = lambda _:_/np.linalg.norm(_)


class ViewTransform(Transform):
    """
    Camera frame conventions,

    #. camera at origin, looking down -Z

    +Y out of page, right handed system::

                      -X 
                       .
                       .
                       .                             |
                       .           |    objects      |
           +Z ---------e . . . . . | . . . l . . . . | . . . . -Z
                       |           |                 |
                       |                             |
                       |
                       |          -near            -far
                       |
                      +X


    ViewTransform transforms from world to camera frame

             eye_c = VT * eye    ( at origin of camera frame )
             look_c = VT * look  ( along -Z of camera frame )
             up_c   = VT * up

    To check the VT 

         VT^I * eye_c = eye_w

    When using random look, eye and up and looking at up_c 
    it turns out that its always has X=0, ie somewhere in Y-Z 

    """ 
    def __init__(self, eye=(0,0,0), look=(0,0,-1), up=(0,1,0) ):
        """
        :param eye:   (x,y,z) position of camera/eye in world frame
        :param look:  (x,y,z) position of target in world frame
        :param up:  
        """
        Transform.__init__(self)
        self.set('eye',np.array(eye))
        self.set('look',np.array(look))
        self.set('up',np.array(up))

    def copy(self):
        return ViewTransform(self.eye,self.look,self.up)

    def dump(self):
        print "eye  %s [%s] " % ( self.eye , np.linalg.norm(self.eye))
        print "look %s [%s] " % ( self.look, np.linalg.norm(self.look))
        print "gaze %s [%s] " % ( self.gaze, np.linalg.norm(self.gaze))
        print "up   %s [%s] " % ( self.up  , np.linalg.norm(self.up))

    # unit vectors for the camera
    gaze = property(lambda self:self.look - self.eye)
    right   = property(lambda self:norm_(np.cross(self.forward, self.up)))   # +X 
    top     = property(lambda self:norm_(np.cross(self.right,self.forward))) # +Y 
    forward = property(lambda self:norm_(self.gaze))                         # -Z
 
    def _calculate_matrix(self):
        """ 
        Construct matrix using the normalized basis vectors::    

                             -Z
                       +Y    .  
                        |   .
                  EY    |  .  -EZ forward 
                  top   | .  
                        |. 
                        E-------- +X
                       /  EX right
                      /
                     /
                   +Z

        """
        r = np.identity(4)
        r[:3,0] = self.right
        r[:3,1] = self.top
        r[:3,2] = -self.forward    # forward is -Z, so negate
  
        t = np.identity(4)
        t[:3,3] = -self.eye[0:3]

        m = np.dot(r.T,t)   
        return m 
def test_viewtransform_canonical():
    """
    when the arbitrary eye and look positions in the world frame correspond 
    to those of the conventional camera frame, get identity
    """
    canonical = ViewTransform(eye=(0,0,0),look=(0,0,-1),up=(0,1,0))
    assert np.allclose( canonical.matrix, np.identity(4) ),   ( canonical )


def test_viewtransform():
    for _ in range(100):
        eye  = (np.random.random(3) - 0.5)*10.
        look = (np.random.random(3) - 0.5)*10.
        up =   (np.random.random(3) - 0.5)*10.
        check_viewtransform( eye, look, up)
        check_world_to_camera_consistency( eye, look, up)

def check_viewtransform(eye, look, up):
    """
    Checks that the ViewTransform when applied to the 
    eye and look positions yields expectations in camera frame 

    """
    vt = ViewTransform(eye=eye,look=look,up=up)
    #print vt

    # transform eye and look coordinates and up direction from world to camera frame
    eye_c = vt(eye)
    look_c = vt(look)
    up_c = vt(up,w=0)      

    xeye_c = np.array((0,0,0,1))      # expect origin in camera frame
    assert np.allclose( eye_c , xeye_c ),  (eye_c, xeye_c)     

    d = np.linalg.norm( look - eye )  # distance between eye and look 
    xlook_c = np.array((0,0,-d,1))    # look should be at Z=-d 
    assert np.allclose( look_c , xlook_c ),  (look_c, xlook_c)     

    # use inverse matrix to transform from camera frame back to world frame 
    eye_w = vt(eye_c, inverse=True)
    look_w = vt(look_c, inverse=True)
    up_w = vt(up_c, w=0, inverse=True)   # using w=0 effectively ignores the translation portion

    assert np.allclose( eye_w[:3] , eye)
    assert np.allclose( look_w[:3] , look)
    assert np.allclose( up_w[:3] , up )

    #print "eye", eye
    #print "eye_c", eye_c
    #print "eye_w", eye_w 

    #print "look", look
    #print "look_c", look_c
    #print "look_w", look_w 
        
    #print "up", up
    #print "up_c", up_c
    #print "up_w", up_w 
 

def check_world_to_camera_consistency(eye, look, up):
    vt = ViewTransform(eye=eye,look=look,up=up)
    from world_to_camera import world_to_camera
    cw = world_to_camera( eye, look, up)
    #print cw
    assert np.allclose( vt.matrix, cw )


def tests():
    test_viewtransform_canonical()
    test_viewtransform() 


if __name__ == '__main__':

    tests()

    eye = np.array((1,0,1))
    look = np.array((2,0,2))
    up = np.array((0,1,0))



 


  
 
