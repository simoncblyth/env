#!/usr/bin/env python

import math
import numpy as np
import glumpy as gp  
from glumpy.trackball import _q_add, _q_normalize, _q_rotmatrix, _v_cross, _v_sub, _v_length, _q_from_axis_angle
from OpenGL.GL import GLfloat

# **Keep this for maintaining attitude, avoid anymore OpenGL usage**


class DAETrackball(gp.Trackball):
    """

    Original glumpy Trackball operation
    -------------------------------------

    State:

    #. `_rotation` orientation quaternion (`_matrix` is always derived from this)
    #. `_x` `_y` (changed by pan_to, actually bug in original)
    #. `_distance`  (no change interface, only via constructor argument)
    #. `_zoom` (changed by zoom_to, clamped .25:10)

       * linear scaling applied to left,right,top,bottom (independant of fixed "aperture" of 35 degrees)

    `_set_orientation(self, theta, phi)`
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    * computes quaternions xrot (from theta) and zrot (from phi)
    * xrot and zrot quats are multiplied **setting** `self._rotation` quat
    * `self._matrix` is calculated from `self._rotation` quat

    This is invoked from `__init__` establising initial orientation from the 
    `theta` `phi` arguments. Although setters for `theta` and `phi` are present 
    which also invoke `_set_orientation` these are never used in normal trackball operation

    `drag_to`
    ~~~~~~~~~ 

    Uses `_rotate` to convert mouse inputs into a quat q
    which is used to change self._rotation by _q_add (actually quat multiplicaton) 
    which then yield the matrix::

            175         self._rotation = _q_add(q,self._rotation)
            ...
            180         m = _q_rotmatrix(self._rotation)
            181         self._matrix = (GLfloat*len(m))(*m)

    `_rotate`
    ~~~~~~~~~

    * mouse/trackpad inputs (x,y,z(x,y)) and (x+dx,y+dy,z(x+dx,y+dy))  
      where z obtained by projection onto virtual trackball 
    * these two 3D points converted into axis and angle
    * axis and angle converted into quat 
    
    `push`
    ~~~~~~~

    GL_PROJECTION matrix from
  
    * aspect and fixed aperture, near, far 
    * _zoom

    GL_MODELVIEW matrix from 

    * `_x` `_y` `_distance`
    * `_matrix`


    Changes made
    ------------

    #. renamed `_distance` to `_z`    
    #. adopt variable `yfov`, discard `zoom`



    """
    def __init__(self, thetaphi=(0,0), xyz=(0,0,3), trackballradius=0.8, translatefactor=1000. ):
        """
        :param thetaphi:
        :param xyz: 3-tuple position, only used in non-lookat mode 
        :param yfov:

        Keep trackball for GL_PROJECTION relevant stuff, not GL_MODELVIEW ?
        """
        self._rotation = [0,0,0,1]
        self._matrix = None
        theta, phi = thetaphi
        self._set_orientation(theta,phi)

        self._x =  xyz[0]
        self._y =  xyz[1]
        self._z =  xyz[2]

        self._count = 0 
        self._RENORMCOUNT = 97
        #self._TRACKBALLSIZE = radius
        self.trackballradius = trackballradius
        self.translatefactor = translatefactor    


        # vestigial
        self.zoom = 0    
        self.distance = 0 


    def home(self):
        self._x = 0.
        self._y = 0.
        self._z = 0.
        self._set_orientation(0,0)

    def __repr__(self):
        return "T %4.1f/%4.1f/%4.1f %3.1f/%3.1f" % \
            ( self._x, self._y, self._z, self.theta, self.phi )
    __str__ = __repr__


    def _get_xyz(self):
        return np.array([self._x, self._y, self._z])    # remove z sign flip
    xyz = property(_get_xyz)


    def drag_to (self, x, y, dx, dy):
        ''' Move trackball view from x,y to x+dx,y+dy. '''
        q = self._rotate(x,y,dx,dy)
        self._rotation = _q_add(q,self._rotation)
        self._count += 1
        if self._count > self._RENORMCOUNT:
            self._rotation = _q_normalize(self._rotation)
            self._count = 0 
        m = _q_rotmatrix(self._rotation)
        self._matrix = (GLfloat*len(m))(*m)


    def _project(self, r, x, y):
        ''' Project an x,y pair onto a sphere of radius r OR a hyperbolic sheet
            if we are away from the center of the sphere.
        '''

        d = math.sqrt(x*x + y*y)
        if (d < r * 0.70710678118654752440):    # Inside sphere
            z = math.sqrt(r*r - d*d)
        else:                                   # On hyperbola
            t = r / 1.41421356237309504880
            z = t*t / d
        return z


    def _rotate(self, x, y, dx, dy):
        ''' Simulate a track-ball.

            Project the points onto the virtual trackball, then figure out the
            axis of rotation, which is the cross product of x,y and x+dx,y+dy.

            Note: This is a deformed trackball-- this is a trackball in the
            center, but is deformed into a hyperbolic sheet of rotation away
            from the center.  This particular function was chosen after trying
            out several variations.
        '''

        if not dx and not dy:
            return [ 0.0, 0.0, 0.0, 1.0]
        last = [x, y,       self._project(self.trackballradius, x, y)]
        new  = [x+dx, y+dy, self._project(self.trackballradius, x+dx, y+dy)]
        a = _v_cross(new, last)
        d = _v_sub(last, new)
        t = _v_length(d) / (2.0*self.trackballradius)
        if (t > 1.0): t = 1.0
        if (t < -1.0): t = -1.0
        phi = 2.0 * math.asin(t)
        return _q_from_axis_angle(a,phi)

    def zoom_to (self, x, y, dx, dy):
        """
        Zoom trackball by a factor dy 
        Changed from glumpy original _zoom to _distance
        so this is now a translation in z direction
        """
        self._z += dy*self.translatefactor   # former adhoc * -5

    def pan_to (self, x, y, dx, dy):
        ''' Pan trackball by a factor dx,dy '''

        self._x += dx*self.translatefactor    # former adhoc *3
        self._y += dy*self.translatefactor


    def _get_matrix(self):
        return self._matrix
    matrix = property(_get_matrix)


if __name__ == '__main__':
    pass
