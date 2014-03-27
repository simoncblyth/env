#!/usr/bin/env python

import math
import numpy as np
import glumpy as gp  
import OpenGL.GL as gl

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
    def __init__(self, thetaphi=(0,0), xyz=(0,0,3), yfov=50, near=0.01, far=10. , nearclip=(1e-5,1e6), farclip=(1e-5,1e6)): 
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
        self._TRACKBALLSIZE = 0.8 

        self._yfov = yfov
        self._near = near
        self._far = far

        self.yfovclip = 1.,179.   # extreme angles are handy in parallel projection
        self.nearclip = nearclip
        self.farclip = farclip


        # vestigial
        self.zoom = 0    
        self.distance = 0 


    def home(self):
        self._x = 0.
        self._y = 0.
        self._z = 0.
        self._set_orientation(0,0)

    def __repr__(self):
        return "yfov %3.1f near %10.5f far %4.1f x %4.1f y %4.1f z%4.1f theta %3.1f phi %3.1f" % \
            ( self._yfov, self._near, self._far, self._x, self._y, self._z, self.theta, self.phi )
    __str__ = __repr__



    def _get_height(self):
        viewport = gl.glGetIntegerv(gl.GL_VIEWPORT)
        return float(viewport[3])
    height = property(_get_height)

    def _get_width(self):
        viewport = gl.glGetIntegerv(gl.GL_VIEWPORT)
        return float(viewport[2])
    width = property(_get_width)

    def _get_aspect(self):
        viewport = gl.glGetIntegerv(gl.GL_VIEWPORT)
        return float(viewport[2])/float(viewport[3])
    aspect = property(_get_aspect)




    def zoom_to (self, x, y, dx, dy):
        """
        Zoom trackball by a factor dy 
        Changed from glumpy original _zoom to _distance
        so this is now a translation in z direction
        """
        self._z += -5*dy/self.height

    def pan_to (self, x, y, dx, dy):
        ''' Pan trackball by a factor dx,dy '''
        self._x += 3*dx/self.width
        self._y += 3*dy/self.height





    def near_to (self, x, y, dx, dy):
        ''' Change near clipping '''
        self.near += self.near*dy/self.height

    def far_to (self, x, y, dx, dy):
        ''' Change far clipping '''
        self.far += self.far*dy/self.height

    def yfov_to (self, x, y, dx, dy):
        ''' Change yfov '''
        self.yfov += 50*dy/self.height


    def _get_near(self):
        return self._near
    def _set_near(self, near):
        self._near = np.clip(near, self.nearclip[0], self.nearclip[1])
    near = property(_get_near, _set_near)

    def _get_far(self):
        return self._far
    def _set_far(self, far):
        self._far = np.clip(far, self.farclip[0],self.farclip[1])
    far = property(_get_far, _set_far)

    def _get_yfov(self):
        return self._yfov
    def _set_yfov(self, yfov):
        self._yfov = np.clip(yfov,self.yfovclip[0],self.yfovclip[1])
    yfov = property(_get_yfov, _set_yfov)


    def _get_matrix(self):
        return self._matrix
    matrix = property(_get_matrix)

    def _get_lrbtnf(self):
        """
        ::

                   . | 
                .    | top 
              +------- 
                near |
                     |
                   
        """
        aspect = self.aspect
        near = self._near  
        far = self._far    
        top = near * math.tan(self._yfov*0.5*math.pi/180.0)  
        bottom = -top
        left = aspect * bottom
        right = aspect * top 

        return np.array([left,right,bottom,top,near,far]) 

    lrbtnf = property(_get_lrbtnf)



if __name__ == '__main__':
    pass
