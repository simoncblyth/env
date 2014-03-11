#!/usr/bin/env python
import math
import numpy as np
from env.graphics.transformations.transformations import Arcball

_EPS = np.finfo(float).eps * 4.0

def angle_axis_from_quaternion( quaternion ):
    """ 
    :param quaternion: normalised quaternion
    :return angle, axis:

    * http://en.wikipedia.org/wiki/Axis-angle_representation
    * http://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToAngle/
    * http://www.lfd.uci.edu/~gohlke/code/transformations.py.html

    """ 
    qw = quaternion[0]
    angle = 2.0*math.acos(qw) 
    den = np.sqrt(1.0-qw*qw)
    if den > _EPS:
        axis = quaternion[1:]/den
    else:
        axis = quaternion[1:]  # meaningless axis ? 
    return angle, axis 

class MyArcball(Arcball):
    @classmethod 
    def make(cls, size, axis1, axis2, constrain=True):
        """
        :param size: window size in pixels
        :param axis1: coordinate axes
        :param axis2: coordinate axes
        :return ball:  Arcball instance with virtual ball centered on the screen 
        """
        ball = cls()
        center = np.array(size)/2.
        radius = np.linalg.norm(center)/2 
        ball.place( center, radius ) 
        ball.setaxes( axis1, axis2 )
        ball.constrain = constrain
        return ball

    def __init__(self, *args, **kwa):
        super(MyArcball, self).__init__(*args, **kwa)

    def angle_axis(self):
        return angle_axis_from_quaternion( self._qnow )


if __name__ == '__main__':

    size = np.array([ 640, 480 ])


    axis1 = np.array([0,0,1])
    axis2 = np.array([1,0,0])


    start = np.array([0.25, 0.25])

    checks = [   [start, np.array([0.25,0.3])], 
                 [start, np.array([0.3,0.25])],
                 [start, np.array([0.4,0.4])],
                 [start, np.array([0.4,0.6])],
                 [start, np.array([0.6,0.4])],
                 [start, np.array([0.6,0.6])],
                 [start, np.array([0.6,0.5])],
               ]

    ball = MyArcball.make( size, axis1, axis2 )
    for down, drag in checks:
        ball.down( size*down )
        ball.drag( size*drag )
        angle, axis = ball.angle_axis()
        print "down %s drag %s angle %s axis %s " % (down, drag, angle, axis )




