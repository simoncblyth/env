#!/usr/bin/env python
"""
"""
import numpy as np
from transform import Transform

from env.graphics.transformations.transformations import \
     quaternion_slerp, \
     quaternion_matrix


class InterpolateTransform(Transform):
    """
    Screen space that the perspective transform ends up in 
    is a funny non-linear space involving perspective division by "z" this
    may be why interpolation not working ?
   
    * http://www.arcsynthesis.org/gltut/Texturing/Tut14%20Interpolation%20Redux.html

    """
    def __init__(self, start_transform, end_transform, fraction=0. , spin=0, shortestpath=True ): 
        """
        :param start_transform: instance of Transform subclass 
        :param end_transform: instance of Transform subclass 
        :param fraction: fraction of the way from start to end

        For definiteness, only the fraction is allowed to change after instantiation
        """
        Transform.__init__(self) 

        self.start_transform = start_transform
        self.end_transform = end_transform

        self.start_position = start_transform.matrix[:3,3]
        self.end_position = end_transform.matrix[:3,3]
 
        self.setFraction(fraction)
     
        self.spin = spin
        self.shortestpath = shortestpath

    def setFraction(self, fraction):
        self.set('fraction', fraction)
        return self

    def __repr__(self):
         return "\n".join(["Interpolate fraction %s " % self.fraction, str(self.start_transform), str(self.end_transform), str(Transform.__repr__(self))]) 

    def copy(self):
        return InterpolateTransform(self.start_transform, self.end_transform , self.fraction, self.spin, self.shortestpath )

    position = property(lambda self:self.start_position*(1.-self.fraction) + self.fraction*self.end_position )  

    def _calculate_matrix(self):

        # from 0 to 1 
        f = self.fraction

        # in conversion to a quaternion, all translation information is dropped
        a_quaternion = self.start_transform.quaternion
        b_quaternion = self.end_transform.quaternion
        f_quaternion = quaternion_slerp(a_quaternion, b_quaternion, f, spin=self.spin, shortestpath=self.shortestpath )

        f_matrix = quaternion_matrix( f_quaternion )   
        f_position = self.start_position*(1.-f) + f*self.end_position
        
        # separately interpolate the translation and shove it into the matrix ???
        f_matrix[:3,3] = f_position

        #t = np.identity(4)
        #t[:3,3] = f_position  
        #f_matrix = np.dot(t, f_matrix ) 

        return f_matrix

    def check_endpoints(self):
        f = self.fraction

        self.setFraction(0)
        assert np.allclose( self.matrix, self.start_transform.matrix ) , (self.matrix, self.start_transform.matrix )
        self.setFraction(1)
        assert np.allclose( self.matrix, self.end_transform.matrix ) , (self.matrix, self.end_transform.matrix )

        self.setFraction(f)



def test_interpolate_transform():

    from view_transform import ViewTransform
 
    X = np.array((1,0,0))
    Y = np.array((0,1,0))
    Z = np.array((0,0,1))
    O = np.array((0,0,0))

    up = Y
    eye = O

    vx = ViewTransform( eye, 10*X, up )
    vz = ViewTransform( eye, 10*Z, up )
    it = InterpolateTransform(vx, vz )

    assert np.allclose( it.setFraction(0).matrix , vx.matrix )
    assert np.allclose( it.setFraction(1).matrix , vz.matrix )

    #for f in np.linspace(0,1,5):
    #    print it.setFraction(f)




if __name__ == '__main__':
    pass
    test_interpolate_transform()

