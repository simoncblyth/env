#!/usr/bin/env python

import math
import numpy as np
import glumpy as gp  
from glumpy.trackball import _q_add, _q_normalize, _q_rotmatrix, _v_cross, _v_sub, _v_length, _q_from_axis_angle
from daeutil import translate_matrix


#from OpenGL.GL import GLfloat

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

    def dump(self):
        print " trackballradius %s " % self.trackballradius
        print " translatefactor %s " % self.translatefactor
        print " _rotation(xyzw) %s " % repr(self._rotation) 


    def _get_xyz(self):
        return np.array([self._x, self._y, self._z])   
    def _set_xyz(self, xyz):
        self._x, self._y, self._z = xyz
    xyz = property(_get_xyz, _set_xyz)

    def _get_translate(self):
        return translate_matrix(self.xyz)   
    translate = property(_get_translate)

    def _get_untranslate(self):
        return translate_matrix(-self.xyz) 
    untranslate = property(_get_untranslate)

    def _get_rotation(self):
        return np.array( self._matrix, dtype=float).reshape(4,4).T   # transposing to match GL_MODELVIEW
    rotation = property(_get_rotation)


    def test_drag_to(self):
        self.drag_to(0,0,0.01,0)

    def drag_to (self, x, y, dx, dy):
        """
        Move trackball view from x,y to x+dx,y+dy. 

        Removed ctypes diddling the _matrix as on converting to numpy array get::
 
            RuntimeWarning: Item size computed from the PEP 3118 buffer format string does not match the actual item size.
        """
        q = self._rotate(x,y,dx,dy)

        drag = x,y,dx,dy
        #print "drag_to drag  %s " % repr(drag)
        #print "drag_to q     %s " % repr(q)   
        #print "drag_to _rot0 %s " % repr(self._rotation)   

        self._rotation = _q_add(q,self._rotation)

        #print "drag_to _rot1  %s " % repr(self._rotation)   

        self._count += 1
        if self._count > self._RENORMCOUNT:
            self._rotation = _q_normalize(self._rotation)
            self._count = 0 
        m = _q_rotmatrix(self._rotation)
        #self._matrix = (GLfloat*len(m))(*m)   # ctyles diddling moved to daeframehandler
        self._matrix = m


    def _set_orientation(self, theta, phi):
        ''' Computes rotation corresponding to theta and phi. ''' 

        self._theta = theta
        self._phi = phi 

        angle = self._theta*(math.pi/180.0)
        sine = math.sin(0.5*angle)
        xrot = [1*sine, 0, 0, math.cos(0.5*angle)]

        angle = self._phi*(math.pi/180.0)
        sine = math.sin(0.5*angle);
        zrot = [0, 0, sine, math.cos(0.5*angle)]

        self._rotation = _q_add(xrot, zrot)
        m = _q_rotmatrix(self._rotation)
        #self._matrix = (GLfloat*len(m))(*m)
        self._matrix = m




    def _project(self, r, x, y):
        """ 
        Project an x,y pair onto a sphere of radius r OR a hyperbolic sheet
        if we are away from the center of the sphere.

        For points inside xy circle::

                  d^2 = x^2 + y^2

                    d < r / sqrt(2)   

                  d^2 < r^2 / 2 

            x^2 + y^2 < r^2 / 2 
   

        determine z from::

                z^2 + d^2 = r^2 

        So are projecting onto sphere the center of which is on the screen plane.
        """
        d = math.sqrt(x*x + y*y)
        if (d < r * 0.70710678118654752440):    
            z = math.sqrt(r*r - d*d)            
        else:                                  
            t = r / 1.41421356237309504880
            z = t*t / d
        return z


    def test_rotate(self):
        drag = (0,0,0.01,0)
        q = self._rotate(*drag)
        print " x,y,dx,dy %s => %s " % (repr(drag), repr(q)) 


    def _rotate(self, x, y, dx, dy):
        """
        Simulate a track-ball.

        Project the points onto the virtual trackball, then figure out the
        axis of rotation, which is the cross product of x,y and x+dx,y+dy.

        Note: This is a deformed trackball-- this is a trackball in the
        center, but is deformed into a hyperbolic sheet of rotation away
        from the center.  This particular function was chosen after trying
        out several variations.
        """
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

        q = _q_from_axis_angle(a,phi)

        #print "_rotate p0   %s " % repr(last)
        #print "_rotate p1   %s " % repr(new)
        #print "_rotate axis %s " % repr(a)
        #print "_rotate t %s phi %s " % (t, phi)
        #print "_rotate q %s  " % repr(q)

        return q

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
    tb = DAETrackball()
    tb.dump()
    #tb.test_rotate()
    tb.test_drag_to()



