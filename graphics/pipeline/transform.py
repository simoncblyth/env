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
     angle_axis_from_quaternion, \
     quaternion_multiply



def qrepr(q):
    angle, axis = angle_axis_from_quaternion(q)
    return  "%s norm %s angle %s axis %s" % ( q, q.dot(q), 180.*angle/math.pi, axis) 


class Transform(object):
    """
    Subclasses need to:
 
    #. store parameters of their transforms
    #. provide argumentless _calculate_matrix method
    
    """
    def __init__(self): 
        self._matrix = None
        self._quaternion = None
        self._pquaternion = None
        self.dirty = True

    def __repr__(self):
        return "%s %s\n%s\n" % ( self.__class__.__name__, qrepr(self.quaternion), self.matrix ) 

    def _calculate_matrix(self):
        raise Exception("subclasses of Transform need to implement _calculate_matrix ")

    def _get_matrix(self):
        if self.dirty or self._matrix is None:
            self._matrix = self._calculate_matrix()
        return self._matrix
    matrix = property(_get_matrix)

    def _get_inverse(self):
        if hasattr(self, '_calculate_matrix_inverse'):
            m = self._calculate_matrix_inverse()
        else: 
            m = invert_homogenous(self.matrix)  # this dont work for scaled homogenous
        return m  
    inverse = property(_get_inverse)

    def _get_quaternion(self):
        if self._quaternion is None or self.dirty:
            self._quaternion = quaternion_from_matrix( self.matrix, isprecise=False )
        return self._quaternion
    quaternion = property(_get_quaternion)

    def _get_pquaternion(self):
        if self._pquaternion is None or self.dirty:
            self._pquaternion = quaternion_from_matrix( self.matrix, isprecise=True )
        return self._pquaternion
    pquaternion = property(_get_pquaternion)


    def set(self, name, value):
        """
        Setter that invalidates the matrix and quaternion 
        whenever anything is set
        """
        self.dirty = True
        self.__dict__[name] = value

    def transform_vertex(self, vert, w=1, inverse=False  ):
        assert vert.shape == (3,) , vert.shape
        vert = np.append(vert, w )
        if inverse:
            m = self.inverse
        else:
            m = self.matrix

        p = np.dot( m, vert )

        # no homogenous division when dealing with directions 
        if w == 1:
            p /= p[3]
     
        return p 
 
    def transform_vertices(self, verts, w=1, inverse=False):
        """
        :param verts: numpy 2d array of vertices, for example with shape (1000,3)

        Extended homogenous matrix multiplication yields (x,y,z,w) 
        which corresponds to coordinate (x/w,y/w,z/w)  
        This is a trick to allow to represent the "division by z"  needed 
        for perspective transforms with matrices, which normally cannot 
        represent divisions.
 
        Steps:

        #. add extra column of ones, eg shape (1000,4)
        #. matrix pre-multiply the transpose of that 
        #. divide by last column  (xw,yw,zw,w) -> (xw/w,yw/w,zw/w,1) = (x,y,z,1)  whilst still transposed
        #. return the transposed back matrix 

        To do the last column divide while not transposed could do::

            (verts.T/verts[:,(-1)]).T


        Discussion on delaying the homogenous divide, to allow working 
        with points at infinity where w=0

        * http://gaim.umbc.edu/2010/06/10/homogeneous-fun/

        """
        assert verts.shape[-1] == 3 and len(verts.shape) == 2, ("unexpected shape", verts.shape )
        if w == 1:
            v = np.concatenate( (verts, np.ones((len(verts),1))),axis=1 )  # add 4th column of ones 
        else:
            v = np.concatenate( (verts, np.zeros((len(verts),1))),axis=1 )  # add 4th column of zeros 

        if inverse:
            m = self.inverse
        else:
            m = self.matrix

        vt = np.dot( m, v.T )

        # no homogenous division when dealing with directions ?
        if w == 1:
            if (vt[-1] == 0).any():
                log.warn("cannot homogenous divide due to zeros %s " % vt ) 
            else: 
                vt /= vt[-1]   

        return vt.T
 
    def __call__(self,v, w=1, inverse=False):
        """
        :param v: single coordinate or list of coordinates, where coordinates are expressed as 3-tuples
        :param w: w should be 1 for points and 0 for directions
        :param inverse:

        Apply the Transform or its inverse
        """
        v = np.array(v)
        if len(v.shape) == 1:
            return self.transform_vertex(v, w=w, inverse=inverse)
        elif len(v.shape) == 2:
            return self.transform_vertices(v, w=w, inverse=inverse)
        else:
            assert 0, ("unexpected shape", v.shape)


    def dump_vertices(self, vertices):
        """
        For debugging vertices that fail to appear  
        """
        points = self(vertices)            # apply all transfomations in one go 
        bounds_ = lambda _:np.min(_, axis=0), np.max(_, axis=0)
        print bounds_(points) 






class ScaleTransform(Transform):
    def __init__(self, scale ):
        Transform.__init__(self)
        self.set('scale',scale)
    def _calculate_matrix(self):
        s = np.identity(4)
        s[0,0] = self.scale
        s[1,1] = self.scale
        s[2,2] = self.scale
        return s

class TranslateTransform(Transform):
    def __init__(self, translate ):
        Transform.__init__(self)
        self.set('translate',translate)
    def _calculate_matrix(self):
        t = np.identity(4)
        t[:3,3] = self.translate 
        return t





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


def invert_scale( s ):
    i = np.identity(4)
    i[0,0] = 1./s[0,0]
    i[1,1] = 1./s[1,1]
    i[2,2] = 1./s[2,2]
    return i   


def scale_matrix( scale ):
    s = np.identity(4)
    s[0,0] = scale
    s[1,1] = scale
    s[2,2] = scale
    return s
 
def test_invert_scale():
    s = scale_matrix(10)
    i = invert_scale( s )
    assert np.allclose(np.dot(s,i), np.identity(4)) , (s,i)       
    assert np.allclose(np.dot(i,s), np.identity(4)) , (s,i)       
    


def check_invert_homogenous_scaled():
    m = random_homogenous_matrix()
    n = invert_homogenous(m)

    s = scale_matrix(10)
    i = invert_scale(s)

    sm = np.dot(s, m)
    print sm

    ni = np.dot(n, i)
    print ni

    print n.dot(m)

    assert np.allclose(i.dot(s), np.identity(4)) , (s,i)       
    assert np.allclose(s.dot(i), np.identity(4)) , (s,i)       

    assert np.allclose(n.dot(m), np.identity(4)) , (s,i)       
    assert np.allclose(m.dot(n), np.identity(4)) , (s,i)       


 


if __name__ == '__main__':
    pass
    test_invert_homogenous()
    test_invert_scale()
    
    check_invert_homogenous_scaled()


