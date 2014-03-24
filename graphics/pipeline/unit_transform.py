#!/usr/bin/env python


import logging
log = logging.getLogger(__name__)

import numpy as np
import math
from transform import Transform, ScaleTransform, TranslateTransform




class KeyView(object):
    def __init__(self, eye, look, up, unit, name=""):
        self.eye = eye
        self.look = look
        self.up = up
        self.unit = unit 
        self.name = name

    _eye  = property(lambda self:self.unit(self.eye))
    _look = property(lambda self:self.unit(self.look))
    _up   = property(lambda self:self.unit(self.up,w=0))
    _eye_look_up = property(lambda self:(self._eye, self._look, self._up)) 

    def __repr__(self):
        #with printoptions(precision=3, suppress=True, strip_zeros=False):
        return "\n".join([
                    "%s %s " % (self.__class__.__name__, self.name),
                    "p_eye  %s eye  %s " % (self.eye,  self._eye),
                    "p_look %s look %s " % (self.look, self._look),
                    "p_up   %s up   %s " % (self.up,   self._up),
                      ])




class UnitTransform(Transform):
    """
    Transforms between input parameter frame coordinates 
    eg `eye=(1,1,1)` `look=(0,0,0)` and the world frame 

    The input parameter frame coordinates correspond to a particular mesh  
    of current interest and are calulated based upon the world frame 
    upper and lower bounds of the mesh vertices.  
    The mean of the bounds is taken as the center of interest and 
    the extent of interest is taken from the maximum linear difference
    of the bounds.  

    This allows the same viewpoint specifation to be applicable to all volumes.
    """
    def __init__(self, bounds=((-1,-1,-1),(1,1,1)), diagonal=False ):
        """
        :param bounds: 2-tuple of upper, lower bounds 
        """
        Transform.__init__(self)
        assert len(bounds) == 2 
        self.set('bounds',np.array(bounds))
        self.set('diagonal',diagonal)
    
    def _get_extent(self):
        """
        #. diagonal extent introduces less intuitive 3D sqrt and squares of sides
        #. linear one dimensional is intuitive
        """
        lower, upper = self.bounds 
        if self.diagonal:
            extent = np.linalg.norm(upper-lower)/2.
        else:
            extent = np.max(upper-lower)/2.     
        return extent

    extent = property(_get_extent)
    center = property(lambda self:np.mean(self.bounds, axis=0))

    def _calculate_matrix(self):
        """
        First scale then translate, because the center is 
        known in the destination frame coordinates.
        """
        s = ScaleTransform(self.extent)
        t = TranslateTransform(self.center)
        return np.dot(t.matrix, s.matrix)

    def _calculate_matrix_inverse(self):
        """
        Untranslate then unscale
        """
        t = TranslateTransform(-self.center)
        s = ScaleTransform(1./self.extent)
        return np.dot(s.matrix, t.matrix)

    def __repr__(self):
        return "\n".join(
               ["UnitTransform",
                "   upper  %s " % self.bounds[1],
                "   center %s " % self.center,
                "   lower  %s " % self.bounds[0],
                "   extent %s " % self.extent, 
                Transform.__repr__(self)])

    def copy(self):
        return UnitTransform(self.bounds)



    
LOWER_P = (-1,-1,-1)
UPPER_P = (1,1,1)

def test_diagonal():
    side = 100 
    lower, upper= (-side,-side,-side), (+side,+side,+side)
    ut = UnitTransform( (lower,upper), diagonal=True )
    m = ut.matrix

    expect = math.sqrt(3.*math.pow(side,2))
    assert ut.extent == expect 
    assert np.allclose(m[:3,:3], np.identity(3)*expect ) 


def test_symmetric():
    side = 100 
    lower, upper= (-side,-side,-side), (+side,+side,+side)
    ut = UnitTransform( (lower,upper), diagonal=False )
    m = ut.matrix

    lower_u = ut(LOWER_P)
    lower_x = side*np.array(LOWER_P)
    assert np.allclose( lower_u[:3], lower_x )

    upper_u = m.dot(np.append(UPPER_P,1))
    upper_x = side*np.array(UPPER_P)
    assert np.allclose( upper_u[:3], upper_x )


def test_3():
    side = 100 
    lower, upper= (0,0,0), (+2*side,+2*side,+2*side)
    ut = UnitTransform( (lower,upper), diagonal=False )
    print "test_3\n",ut

    lower_u = ut(LOWER_P)

    #print LOWER_P
    #print lower_u
    #print lower
    assert np.allclose( lower_u[:3], np.array(lower)), (LOWER_P,lower_u, lower)

    upper_u = ut(UPPER_P)
    assert np.allclose( upper_u[:3], np.array(upper)), (UPPER_P,upper_u, upper)


def test_inverse():
   
    side = 100 
    lower, upper= (0,0,0), (+2*side,+2*side,+2*side)
    ut = UnitTransform( (lower,upper), diagonal=False )

    m = ut.matrix
    i = ut.inverse

    assert np.allclose( np.identity(4), np.dot(m,i) )
    assert np.allclose( np.identity(4), np.dot(i,m) )


if __name__ == '__main__':
    test_diagonal()
    test_symmetric()
    test_3()
    test_inverse()





 


