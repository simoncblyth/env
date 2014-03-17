#!/usr/bin/env python
"""

"""

import numpy as np
from transform import Transform


class OrientationTransform(Transform):
    """
    Nomenclature:

    #. an orientation is a state 
    #. a rotation is an operation, usually changing the state

    """
    def __init__(self, angle=0, axis=(0,0,0),center=(0,0,0)):
        """
        :param angle:
        :param axis"
        :param center:  
        """
        Transform.__init__(self)         
        self.setAttitude( angle, axis, center )
        self._quaternion = None

    def setAttitude( self, angle, axis, center ):
        """
        :param angle: in degrees
        :param axis:
        """
        self.set('angle',angle)
        self.set('axis', np.array(axis))
        self.set('center',np.array(center)) 

    def copy(self):
        return OrientationTransform(self.angle, self.axis, self.center)

    def _get_quaternion(self):
        if self._quaternion is None or self.dirty:
            self._quaternion = quaternion_about_axis( self.angle*math.pi/180., self.axis )        
        return self._quaternion

    quaternion = property(_get_quaternion)

    def _calculate_matrix(self):
        return orient_around_matrix( self.center, self.quaternion )

    def add(self, name, delta):
        init = getattr(self, name)
        if name in ('center',):
            self.set(name, init+np.array(delta))
        else:
            self.set(name, init+delta)


def orient_around_matrix( center, quaternion ):
    """
    :param center:
    :param quaternion:
    """
    pre_orient = translation_matrix( -center )
    orient = quaternion_matrix( quaternion )   
    post_orient = translation_matrix( center )
    return post_orient.dot(orient).dot(pre_orient) 





def test_orientation():
    ot = OrientationTransform()
    print ot.matrix
    print ot.quaternion
    assert np.allclose( ot.matrix, np.identity(4))
    assert np.allclose( ot.quaternion, np.array((1,0,0,0)))

    ot = OrientationTransform(angle=30., axis=(0,1,0), center=(0,0,0) )
    print ot.matrix
    print ot.quaternion



if __name__ == '__main__':
    test_orientation()

