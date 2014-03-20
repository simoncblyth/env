#!/usr/bin/env python
"""
"""
import numpy as np
from transform import Transform, TranslateTransform
from view_transform import ViewTransform

from env.graphics.transformations.transformations import \
     quaternion_slerp, \
     quaternion_matrix, \
     quaternion_multiply, \
     angle_axis_from_quaternion, \
     quaternion_conjugate


X = np.array((1,0,0))
Y = np.array((0,1,0))
Z = np.array((0,0,1))
O = np.array((0,0,0))

 
class InterpolateTransform(Transform):
    animate = True  # mainly for debug, to switch off animation
    def __init__(self, start_transform, end_transform, fraction): 
        Transform.__init__(self) 
        self.start_transform = start_transform
        self.end_transform = end_transform
        self.setFraction(fraction)

        self.start_position = start_transform.matrix[:3,3]
        self.end_position = end_transform.matrix[:3,3]

    def setFraction(self, fraction):
        self.set('fraction', fraction)
        return self

    def interpolated_position(self, f ):
        return self.start_position*(1.-f) + f*self.end_position   
    position = property(lambda self:self.interpolated_position(self.fraction))  

    def check_endpoints(self):
        f = self.fraction

        self.setFraction(0)
        assert np.allclose( self.matrix, self.start_transform.matrix ) , (self.matrix, self.start_transform.matrix )
        self.setFraction(1)
        assert np.allclose( self.matrix, self.end_transform.matrix ) , (self.matrix, self.end_transform.matrix )

        self.setFraction(f)

    def _transition_quaternion_angle_axis(self):
        """
        * http://stackoverflow.com/questions/9953764/getting-rotation-axis-from-initial-and-final-rotated-quaternions
        * http://3dgep.com/?p=1815

        ::

                    q2 = qT * q1
            q2 * q1^-1 = qT * (q1 * q1^-1) = qT

                 => qT = q2 * q1^-1   

                    qT = q2 * q1^-1 = q2 * q1*    # inverse of unit quaternion is its conjugate 

                                                  #  [s, v]* = [s, -v]     conjugate quat has vector part negated 

        """
        qa = self.start_transform.quaternion
        qb = self.end_transform.quaternion  

        qa_conj = quaternion_conjugate(qa)
        qt = quaternion_multiply( qb, qa_conj )

        angle, axis = angle_axis_from_quaternion( qt )
        return qt, angle, axis 

    transition_quaternion_angle_axis = property(_transition_quaternion_angle_axis)


    def dump(self):
         return "\n".join(["Interpolate fraction %s " % self.fraction, str(self.start_transform), str(self.end_transform), str(Transform.__repr__(self))]) 

    def __repr__(self):
         return "\n".join(["%s fraction %s " % (self.__class__.__name__,self.fraction), str(Transform.__repr__(self))]) 


class InterpolateViewTransform(InterpolateTransform):
    """ 
    Simple approach of linear interpolation of the 
    inputs is much easier to understand and control, compared
    to quaternions. Quats by themselves are OK, the trouble 
    is how to combine the interpolated translation with the rotation, while
    keeping the object in the frame.

    This is a perhaps wasteful way of implementing the idea however.
    """
    def __init__(self, start_transform, end_transform, fraction  ):
        InterpolateTransform.__init__(self, start_transform, end_transform, fraction) 

    def interpolated_eye(self, f ):
        return self.start_transform.eye*(1.-f) + f*self.end_transform.eye   
    eye = property(lambda self:self.interpolated_eye(self.fraction))  

    def interpolated_look(self, f ):
        return self.start_transform.look*(1.-f) + f*self.end_transform.look   
    look = property(lambda self:self.interpolated_look(self.fraction))  

    def interpolated_up(self, f ):
        return self.start_transform.up*(1.-f) + f*self.end_transform.up   
    up = property(lambda self:self.interpolated_up(self.fraction))  

    def interpolated_gaze(self, f):
        return self.look - self.eye
    gaze = property(lambda self:self.interpolated_gaze(self.fraction))

    distance = property(lambda self:np.linalg.norm(self.gaze))


    def _calculate_matrix(self):
        view = ViewTransform( self.eye, self.look, self.up )
        return view.matrix



class QuaternionInterpolateTransform(InterpolateTransform):
    """
    Screen space that the perspective transform ends up in 
    is a funny non-linear space involving perspective division by "z" this
    may be why interpolation not working ?
   
    * http://www.arcsynthesis.org/gltut/Texturing/Tut14%20Interpolation%20Redux.html

    """
    def __init__(self, start_transform, end_transform, fraction , center=None, spin=0, shortestpath=True): 
        """
        :param start_transform: instance of Transform subclass 
        :param end_transform: instance of Transform subclass 
        :param fraction: fraction of the way from start to end

        For definiteness, only the fraction is allowed to change after instantiation
        """
        InterpolateTransform.__init__(self, start_transform, end_transform, fraction) 

        if center is None:
            self.center = self.interpolated_position(0.5)
        else:
            self.center = center 
     
        self.spin = spin
        self.shortestpath = shortestpath

    def copy(self):
        return QuaternionInterpolateTransform(self.start_transform, self.end_transform , self.fraction, self.center, self.spin, self.shortestpath )


    def _calculate_matrix(self):

        # from 0 to 1 
        f = self.fraction

        # in conversion to a quaternion, all translation information is dropped
        qa = self.start_transform.quaternion
        qb = self.end_transform.quaternion
        qf = quaternion_slerp(qa, qb, f, spin=self.spin, shortestpath=self.shortestpath )

        mf = quaternion_matrix( qf )   

        ta = TranslateTransform(-self.center).matrix   # center is fixed over interpolation
        tb = TranslateTransform(self.center).matrix 

        f_rotate_about_center = tb.dot(mf).dot(ta)
        
        f_translate = TranslateTransform(self.position).matrix  # position varies with interpolation

        matrix = f_rotate_about_center
        #matrix = f_translate.dot(f_matrix_about_middle)

        matrix[:3,3] = self.position 


        return matrix



def test_interpolate_view_transform():
    up, eye = Y, O
    vx = ViewTransform( eye, 10*X, up )
    vz = ViewTransform( eye, 10*Z, up )
    iv = InterpolateViewTransform(vx, vz )
    assert np.allclose( iv.setFraction(0).matrix , vx.matrix )
    assert np.allclose( iv.setFraction(1).matrix , vz.matrix )

    for f in np.linspace(0,1,5):
        print iv.setFraction(f)





def test_interpolate_transform():
 
    up, eye = Y, O
    vx = ViewTransform( eye, 10*X, up )
    vz = ViewTransform( eye, 10*Z, up )
    it = InterpolateTransform(vx, vz )

  

    assert np.allclose( it.setFraction(0).matrix , vx.matrix )
    assert np.allclose( it.setFraction(1).matrix , vz.matrix )




if __name__ == '__main__':
    pass
    #test_interpolate_transform()
    test_interpolate_view_transform()

