#!/usr/bin/env python
"""
"""
import numpy as np
from world_to_camera import world_to_camera
from camera_to_orthographic import camera_to_orthographic
from orthographic_to_canonical import orthographic_to_canonical
from canonical_to_screen import canonical_to_screen

def world_to_screen( eye, look, up, near, far, left, right, bottom, top, nx, ny, flip=True ):

    cw = world_to_camera(eye, look, up )

    oc = camera_to_orthographic( near, far )

    co = orthographic_to_canonical( left, bottom, near,  right, top, far )

    sc = canonical_to_screen( nx, ny, flip=flip )

    sw = sc.dot(co).dot(oc).dot(cw)

    return sw
    

if __name__ == '__main__':

    nx, ny, flip = 640., 480., True

    eye = (10,10,10)
    look = (20,20,20)
    up = (0,1,0)


    near, far = 5, 100
    bt = 10 

    aspect = ny/nx
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

    print "sw\n", sw


    gaze = np.array(look) - np.array(eye)
    leftright = np.cross(gaze,up)
    bottomtop = np.cross(leftright,gaze)

    eyeline_ = lambda f:np.array(look) + gaze*f   
    leftright_ = lambda f:np.array(look) + leftright*f   
    bottomtop_ = lambda f:np.array(look) + bottomtop*f   

    s_look = t_( sw, look )
    print "look %-15s s_look %-15s " % ( look, s_look )

    # positions along eyeline expected to correspond to the same pixel position

    print "eyeline"
    for f in (0.,0.5,1.,2.,100.):
        pos = eyeline_(f)
        s_pos = t_( sw, pos )
        print "f %s pos %-15s s_pos %-15s " % ( f, pos, s_pos )

    print "leftright"
    for f in (0.,0.5,1.,2.,100.):
        pos = leftright_(f)
        s_pos = t_( sw, pos )
        print "f %s pos %-15s s_pos %-15s " % ( f, pos, s_pos )

    print "bottomtop"
    for f in (0.,0.5,1.,2.,100.):
        pos = bottomtop_(f)
        s_pos = t_( sw, pos )
        print "f %s pos %-15s s_pos %-15s " % ( f, pos, s_pos )


    




