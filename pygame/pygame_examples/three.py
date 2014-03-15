#!/usr/bin/env python
"""
The approach taken here for projections got too messy 
and difficult to debug. 
Instead have plumped for full pipeline monty in env.graphics.pipeline.world_to_screen

Whats left here, is not currently in use.
"""
import math
import numpy as np

class Quaternion(object):
    """
    https://github.com/mrdoob/three.js/blob/master/src/math/Quaternion.js
    """
    @classmethod
    def fromAxisAngle(cls, axis, angle, normalize=False):
        q = cls()
        q.setFromAxisAngle(axis, angle, normalize=normalize)
        return q

    def __init__(self, x = 0, y = 0, z = 0, w = None):
        self.x, self.y, self.z = float(x), float(y), float(z)
        self.w = 1. if w is None else w ;

    def setFromAxisAngle(self, axis, angle, normalize=False):
        """
        http://www.euclideanspace.com/maths/geometry/rotations/conversions/angleToQuaternion/index.htm
        axis have to be normalized
        """
        if normalize:
            axis = axis / np.linalg.norm(axis)

        halfAngle = angle / 2.
        s = math.sin(halfAngle)

        self.x = axis[0] * s
        self.y = axis[1] * s
        self.z = axis[2] * s
        self.w = math.cos(halfAngle)

    def __repr__(self):
        return "%s %s  (%s,%s,%s) " % (self.__class__.__name__, self.w, self.x,self.y,self.z )   

    def applyQuaternion(self, q ):
        """
        https://github.com/mrdoob/three.js/blob/master/src/math/Vector3.js
        """
        x = self.x;
        y = self.y;
        z = self.z;

        qx = q.x;
        qy = q.y;
        qz = q.z;
        qw = q.w;

        # calculate quat * vector

        ix =  qw * x + qy * z - qz * y;
        iy =  qw * y + qz * x - qx * z;
        iz =  qw * z + qx * y - qy * x;
        iw = -qx * x - qy * y - qz * z;

        # calculate result * inverse quat

        self.x = ix * qw + iw * -qx + iy * -qz - iz * -qy;
        self.y = iy * qw + iw * -qy + iz * -qx - ix * -qz;
        self.z = iz * qw + iw * -qz + ix * -qy - iy * -qx;



if __name__ == '__main__':
    pass



 


