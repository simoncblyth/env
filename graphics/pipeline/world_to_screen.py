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
 





