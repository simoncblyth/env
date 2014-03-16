#!/usr/bin/env python
"""

Seealso:

* http://www.opengl.org/sdk/docs/man2/xhtml/glFrustum.xml
* http://www.opengl.org/sdk/docs/man2/xhtml/gluPerspective.xml
* https://pyrr.readthedocs.org/en/latest/
* https://github.com/mrdoob/three.js/blob/master/src/math/Quaternion.js

"""

import logging
log = logging.getLogger(__name__)

import numpy as np
import math

from world_to_camera import world_to_camera
from camera_to_orthographic import camera_to_orthographic
from orthographic_to_canonical import orthographic_to_canonical
from canonical_to_screen import canonical_to_screen

from env.graphics.transformations.transformations import \
     translation_matrix, \
     quaternion_from_matrix, \
     quaternion_matrix, \
     quaternion_about_axis, \
     quaternion_slerp, \
     quaternion_multiply

norm_ = lambda _:_/np.linalg.norm(_)

def world_to_screen_symmetric( eye, look, up, near, far, yfov, nx, ny, flip ):
    """
    ::

                                                  +
                                                  |
                                                  |
                           +                      | x
                           | xs                   |
       Z ------0 . . . . . |. . . . . . . . . . . |
               |    d      |                      |
               |           +                      |
               |         near                     | 
               X                                  | 
                                                  +
                                                 far 
    """
    aspect = float(ny)/float(nx) 
    top = near * math.tan( yfov * 0.5 * math.pi / 180. );
    bottom = -top
    right = top*aspect 
    left = -right

    log.debug("ny %s nx %s aspect %s " % (ny, nx, aspect )) 
    log.debug("near %s yfov %s " % (near, yfov))
    log.debug("top %s bottom %s " % (top, bottom))
    log.debug("right %s left %s " % (right, left))

    return world_to_screen( eye, look, up, near, far, left, right, bottom, top, nx, ny, flip )


def world_to_screen( eye, look, up, near, far, left, right, bottom, top, nx, ny, flip ):
    """
    :return sw: perspective transform matrix, going from world to screen coordinates
    """

    cw = world_to_camera(eye, look, up )

    oc = camera_to_orthographic( near, far )

    co = orthographic_to_canonical( left, bottom, near,  right, top, far )

    sc = canonical_to_screen( float(nx), float(ny), flip  )

    sw = sc.dot(co).dot(oc).dot(cw)

    return sw
    

class Transform(object):
    def __init__(self): 
        self._matrix = None
        self.dirty = True

    def _get_matrix(self):
        if self.dirty or self._matrix is None:
            self._matrix = self._calculate_matrix()
        return self._matrix

    matrix = property(_get_matrix)

    def set(self, name, value):
        """
        Setter that invalidates the matrix and quaternion 
        whenever anything is set
        """
        self.dirty = True
        self.__dict__[name] = value



def orient_around_matrix( center, quaternion ):
    pre_orient = translation_matrix( -center )
    orient = quaternion_matrix( quaternion )   
    post_orient = translation_matrix( center )
    return post_orient.dot(orient).dot(pre_orient) 


class OrientationTransform(Transform):
    """
    Nomenclature:

    #. an orientation is a state 
    #. a rotation is an operation, usually changing the state

    """
    def __init__(self, angle=30, axis=(0,1,0),center=(0,0,0)):
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


class InterpolateTransform(Transform):
    def __init__(self, start_orientation, start_position, end_orientation, end_position, fraction=0. , spin=0, shortestpath=True ): 
        """
        :param start_orientation: instance of OrientationTransform 
        :param start_position:   
        :param end_orientation: instance of OrientationTransform 
        :param end_position:   
        :param fraction: fraction of the way from start to end

        For definiteness, only the fraction is allowed to change after instantiation
        """
        Transform.__init__(self) 

        self.start_orientation = start_orientation
        self.start_position = np.array(start_position)

        self.end_orientation = end_orientation
        self.end_position = np.array(end_position)

        self.spin = spin
        self.shortestpath = shortestpath

        self.set('fraction', fraction)

    def _calculate_matrix(self):

        a_quaternion = self.start_orientation.quaternion
        a_position = self.start_position

        b_quaternion = self.end_orientation.quaternion
        b_position = self.end_position

        f = self.fraction

        f_position = a_position*(1.-f) + f*a_position  
        f_quaternion = quaternion_slerp(a_quaternion, b_quaternion, f, spin=self.spin, shortestpath=self.shortestpath )

        return orient_around_matrix( f_position, f_quaternion )

    def copy(self):
        return InterpolateTransform(self.start_orientation, self.start_position, self.end_orientation, self.end_position , self.fraction )




class PerspectiveTransform(Transform):
    def __init__(self, eye=(0,0,0), look=(1,0,0), up=(0,1,0), near=0.1, far=100, yfov=30, nx=640, ny=480, flip=True ):
        """
        :param eye:   (x,y,z) position of camera/eye in world frame
        :param look:  (x,y,z) position of target in world frame
        :param near:  distance from eye to view frustrum near plane, in world frame dimensions 
                      (should be less than the distance to geometry of interest, but greater than zero, **never zero**)
        :param far:   distance from eye to frustrum far place, in world frame dimensions    
                      (should be somewhat greater than the distance to geometry of interest, but not huge)
        :param yfov:  camera vertical field of view in degress, eg 30, 45, 90
        :param nx:    left-right pixel dimensions
        :param ny:    bottom-top pixel dimensions
        :param flip:  when True makes top left correpond to pixel coordinate (0,0), instead of bottom left
        """
        Transform.__init__(self)
        self.setViewpoint( eye, look, up, float(near), float(far) )
        self.setCamera( yfov, nx, ny, flip )

    def setViewpoint(self, eye, look, up, near, far):
        self.set('eye',np.array(eye))
        self.set('look',np.array(look))
        self.set('up',np.array(up))
        self.set('near',near)
        self.set('far',far)

    def setCamera(self, yfov, nx, ny, flip ):
        self.set('yfov',yfov)
        self.set('nx',nx)
        self.set('ny',ny)
        self.set('flip',flip)

    def copy(self):
        return PerspectiveTransform(self.eye,self.look,self.up,self.near,self.far, self.yfov, self.nx, self.ny, self.flip)

    screensize = property(lambda self:(int(self.nx), int(self.ny)))

    gaze = property(lambda self:self.look - self.eye)

    # unit vectors for the camera
    forward = property(lambda self:norm_(self.gaze))
    right   = property(lambda self:norm_(np.cross(self.forward, self.up)))   # perpendiular to gaze and up, left-right in camera frame
    top     = property(lambda self:norm_(np.cross(self.right,self.forward)))

    def _calculate_matrix(self):
        return world_to_screen_symmetric( self.eye, self.look, self.up, self.near, self.far, self.yfov, self.nx, self.ny, self.flip)

    def add(self, name, delta):
        init = getattr(self, name)
        if name in ('eye','look',):
            self.set(name, init+np.array(delta))
        else:
            self.set(name, init+delta)


def test_world_to_screen():

    nx, ny, flip, debug  = 640., 480., True, True
    aspect = ny/nx

    eye = (10,10,10)
    look = (20,20,20)
    up = (0,1,0)

    near, far = 5, 100      # aim to contain the geometry of interest
    bt = 10 

    bottom, top =  -bt, bt
    left, right =  -bt*aspect, bt*aspect


    sw = world_to_screen( eye, look, up, near, far, left, right, bottom, top, nx, ny, flip=flip)
    def t_( m , v ):
        """
        Extended homogenous matrix multiplication yields 
        (x,y,z,w) which corresponds to coordinate (x/w,y/w,z/w)  
        """
        p = np.dot( m, np.append(v,1))
        p /= p[3]
        return p 

    if debug:
        print "sw\n", sw


    gaze = np.array(look) - np.array(eye)
    leftright = norm_(np.cross(up, gaze))      
    bottomtop = norm_(np.cross(gaze,leftright)) 
    ngaze = norm_(gaze)


    def eyeline_(_):
        """
        World frame parametric line along gaze at

        ::
 
              e----n---l----------f----
                   |              |
                   

        #. _= -1 at near plane
        #. _= +1 at far plane

        """
        r = (f + 1.)*2.                  # 0 to 1
        d = near*(1.-f) + far*f          # near to far 
        return np.array(eye) + ngaze*d   # 

    # lines in plane perpendicular to the gaze, where plane contains the look point
    distance = np.linalg.norm(gaze)    # distance along gaze 
    pcorr = distance/near              # perspective correction to span the screen with f -1,1

    leftright_ = lambda f:np.array(look) + leftright*f*bt*pcorr*aspect   
    bottomtop_ = lambda f:np.array(look) + bottomtop*f*bt*pcorr   

    s_look = t_( sw, look )
    if debug:
        print "look %-15s s_look %-15s " % ( look, s_look )

    f_range = np.arange(-1,1.25,0.25)  # -1 to 1
    fmt = "f %-5s pos %-50s s_pos %-50s " 

    print "eyeline"
    for f in f_range:
        pos = eyeline_(f)
        s_pos = t_( sw, pos )
        print fmt % ( f, pos, s_pos )

    print "leftright"
    for f in f_range:
        pos = leftright_(f)
        s_pos = t_( sw, pos )
        print fmt % ( f, pos, s_pos )

    print "bottomtop"
    for f in f_range:
        pos = bottomtop_(f)
        s_pos = t_( sw, pos )
        print fmt  % ( f, pos, s_pos )


def test_world_to_screen_symmetric():
   
    nx, ny, flip = 640., 480., True

    eye = (10,10,10)
    look = (20,20,20)
    up = (0,1,0)

    near, far = 5, 100      # aim to contain the geometry of interest


    bt = 10 

    aspect = float(ny)/float(nx)
    bottom, top =  -bt, bt
    left, right =  -bt*aspect, bt*aspect

    sw = world_to_screen( eye, look, up, near, far, left, right, bottom, top, nx, ny, flip )
    def t_( m , v ):
        """
        Extended homogenous matrix multiplication yields 
        (x,y,z,w) which corresponds to coordinate (x/w,y/w,z/w)  
        """
        p = np.dot( m, np.append(v,1))
        p /= p[3]
        return p 

    print "sw\n", sw

    yfov = math.atan(top/near)/(0.5 * math.pi/180. ) 
    sws = world_to_screen_symmetric( eye, look, up, near, far, yfov, nx, ny, flip ) 

    print "sws\n", sws

    print "sws-sw\n", sws-sw 
    assert np.allclose(sw,sws), sw-sws


    pt = PerspectiveTransform()
    pt.setViewpoint( eye, look, up, near, far )
    pt.setCamera( yfov , nx, ny, flip )

    psw = pt.matrix
    assert np.allclose(sw,psw), sw-psw

 
if __name__ == '__main__':

    #test_world_to_screen()
    #test_world_to_screen_symmetric()

    logging.basicConfig(level=logging.INFO)

    eye = (10,10,10)
    look = (20,20,20)
    up = (0,1,0)
    near, far = 5, 100      # aim to contain the geometry of interest
    yfov,nx,ny,flip = 30,640,480,True
   
    pt = PerspectiveTransform()
    pt.setViewpoint( eye, look, up, near, far )
    pt.setCamera( yfov , nx, ny, flip )

    print pt.matrix
    verts = np.vstack([look,look,look])
    print verts
    print pt(verts)
 





