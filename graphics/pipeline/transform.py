#!/usr/bin/env python
"""


#. Transform
#. PerspectiveTransform
#. InterpolateTransform


Reference

* http://www.arcsynthesis.org/gltut/Positioning/Tutorial%2007.html

* http://web.engr.oregonstate.edu/~grimmc/content/research/cameraInterp.html
* http://web.engr.oregonstate.edu/~grimmc/content/papers/tog2005ci/tog2005ci.pdf

"""
import logging
log = logging.getLogger(__name__)

import numpy as np
import math


norm_ = lambda _:_/np.linalg.norm(_)

from env.graphics.transformations.transformations import \
     translation_matrix, \
     translation_from_matrix, \
     quaternion_from_matrix, \
     quaternion_matrix, \
     quaternion_about_axis, \
     quaternion_slerp, \
     euler_matrix, \
     quaternion_multiply



class Transform(object):
    """
    Subclasses need to:
 
    #. store parameters of their transforms
    #. provide argumentless _calculate_matrix method
    
    """
    def __init__(self): 
        self._matrix = None
        self._quaternion = None
        self.dirty = True

    def __repr__(self):
        return "%s %s\n%s\n" % ( self.__class__.__name__, self.quaternion, self.matrix ) 

    def _calculate_matrix(self):
        raise Exception("subclasses of Transform need to implement _calculate_matrix ")

    def _get_matrix(self):
        if self.dirty or self._matrix is None:
            self._matrix = self._calculate_matrix()
        return self._matrix
    matrix = property(_get_matrix)

    def _get_quaternion(self):
        if self._quaternion is None or self.dirty:
            self._quaternion = quaternion_from_matrix( self.matrix )
        return self._quaternion
    quaternion = property(_get_quaternion)

    def set(self, name, value):
        """
        Setter that invalidates the matrix and quaternion 
        whenever anything is set
        """
        self.dirty = True
        self.__dict__[name] = value

    def __call__(self, v , w=1, inverse=False):
        """
        :param v: corrdinate expressed as 3-tuple
        :param w: w should be 1 for points and 0 for directions

        Apply the Transform or its inverse
        """
        if inverse:
            m = invert_homogenous(self.matrix)
        else:
            m = self.matrix
        return m.dot(np.append(v[:3],w))


def invert_homogenous( m ):
    """
    http://www.euclideanspace.com/maths/geometry/affine/matrix4x4/

    This is not a general inversion, it makes 
    use of the special properties of rotation-translation matrices

    The matrix multiplication order is critical, 
    opposite order works for rotation only but not for rotation + translation matrices
    """

    r = np.identity(4)
    r[:3,:3] = m[:3,:3]     # rotation portion

    t = np.identity(4)
    t[:3,3] = -m[:3,3]      # negate translation portion

    return np.dot(r.T,t)    # transposed rotation * negated translation 


  
def test_invert_homogenous():
    for _ in range(100):
        check_invert_homogenous()

def random_homogenous_matrix():
    angles = (np.random.random(3) - 0.5) * (2*math.pi)
    m = euler_matrix( angles[0], angles[1],angles[2], "sxyz")
    m[:3,3] = np.random.random(3)*10.
    return m  

def check_invert_homogenous():
    """
    http://www.blancmange.info/notes/maths/vectors/homo/ 
    """
    m = random_homogenous_matrix()
    i = invert_homogenous(m)
    assert np.allclose(np.dot(m,i), np.identity(4)) , (m,i)       
    assert np.allclose(np.dot(i,m), np.identity(4)) , (m,i)       




if __name__ == '__main__':
    pass
    test_invert_homogenous()



