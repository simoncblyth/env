#!/usr/bin/env python


import logging
log = logging.getLogger(__name__)

import numpy as np
import math
from transform import Transform
from unit_transform import UnitTransform

norm_ = lambda _:_/np.linalg.norm(_)


class ViewTransform(Transform):
    """
    A rotation and translation to orient:

    #. eye/camera at origin
    #. looking down -Z

    Definition of up, 

       a vector in the plane bisecting the viewers head 
       into left and right halves and "pointing to the sky"

    (often +Y)


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
        :param eye:   (x,y,z) position of camera/eye 
        :param look:  (x,y,z) position of target 
        :param up:    direction of up

        The eye/look/up parameters are defined in the "volume" frame centered 
        around a volume center. This is very convenient as consistent parameters can be
        applied to multiple volumes and it is easy to pick values to navigate 
        around a volume.

        This ViewTransform however needs to apply to world frame coordinates
        converting them into the camera frame.

        The unit transform is applied to the volume frame input  
        parameters in order to construct this matrix. The unit 
        transform is not applied to the world coordinate "customers" 
        of this transform

        """
        Transform.__init__(self)

        self.set('eye',np.array(eye)[:3])
        self.set('look',np.array(look)[:3])
        self.set('up',np.array(up)[:3])

    def __repr__(self):
        return "\n".join(["eye %s look %s up %s distance %s " % ( self.eye, self.look, self.up, self.distance ), Transform.__repr__(self)])

    def copy(self):
        return ViewTransform(self.eye,self.look,self.up)

    def dump(self):
        print "eye  %s [%s] " % ( self.eye , np.linalg.norm(self.eye))
        print "look %s [%s] " % ( self.look, np.linalg.norm(self.look))
        print "gaze %s [%s] " % ( self.gaze, np.linalg.norm(self.gaze))
        print "up   %s [%s] " % ( self.up  , np.linalg.norm(self.up))

    # unit vectors for the camera
    gaze = property(lambda self:self.look - self.eye)
    distance = property(lambda self:np.linalg.norm(self.gaze))

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
        r[:3,2] = -self.forward  # conventionally forward is -Z, so negate to use right-handed coordinate system
  
        t = np.identity(4)
        translate = self.eye[:3] 
        t[:3,3] = -translate[:3]

        m = np.dot(r.T,t)   
        return m 


def test_viewtransform_canonical():
    """
    when the arbitrary eye and look positions in the world frame correspond 
    to those of the conventional camera frame, expect to 
    get identity matrix for the transform
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


def assert_allclose( result, expect, msg ):
    assert np.allclose( result , expect ),  (result, expect, msg )     

def check_viewtransform(eye, look, up):
    """
    Checks that the ViewTransform when applied to the 
    eye and look positions yields expectations in camera frame 
    """
    vt = ViewTransform(eye=eye,look=look,up=up)

    # transform eye and look coordinates and up direction from world to camera frame
    eye_c = vt(eye)
    look_c = vt(look)
    up_c = vt(up,w=0)      

    assert_allclose( eye_c ,  np.array((0,0,0,1)),            "expect eye to be at origin in camera frame")
    assert_allclose( look_c , np.array((0,0,-vt.distance,1)), "look should be at Z=-d  in camera frame")

    # use inverse matrix to transform from camera frame back to world frame 
    eye_w = vt(eye_c[:3], inverse=True)
    look_w = vt(look_c[:3], inverse=True)
    up_w = vt(up_c[:3], w=0, inverse=True)   # using w=0 effectively ignores the translation portion

    # expect to get back to the inputs
    assert np.allclose( eye_w[:3] , eye)
    assert np.allclose( look_w[:3] , look)
    assert np.allclose( up_w[:3] , up )


def check_world_to_camera_consistency(eye, look, up):
    """
    Checking the two implementations agree
    """ 
    vt = ViewTransform(eye=eye,look=look,up=up)
    from world_to_camera import world_to_camera
    cw = world_to_camera( eye, look, up)
    assert np.allclose( vt.matrix, cw )


def tests():
    test_viewtransform_canonical()
    test_viewtransform() 


if __name__ == '__main__':
    tests()



 


  
 
