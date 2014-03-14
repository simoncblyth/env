#!/usr/bin/env python
"""

PIPELINE
========

#. world frame
#. camera frame
#. 
#. 




"""

import numpy as np

def norm( v ):
    return v/np.linalg.norm(v)

def world_to_camera( eye, look, up ):
    """
    `up` vector, `eye` and `look` positions in world XYZ frame:: 

               up   
                 ^  l
                 | /
           Y     |/ 
           |     e                           
           |                             
           |                               
           *----- X                         
          /                               
         /                               
        Z

    Camera frame UVW basis vectors::

               V
               |
               | 
               |
         W ----e - - l
                \
                 \
                  U

    """
    look = np.array(look)
    eye = np.array(eye)

    gaze = look - eye 

    print "eye  %s [%s] " % ( eye , np.linalg.norm(eye))
    print "look %s [%s] " % ( look, np.linalg.norm(look))
    print "gaze %s [%s] " % ( gaze, np.linalg.norm(gaze))
    print "up   %s [%s] " % ( up  , np.linalg.norm(up))

    # UVW basis vectors for the camera frame

    U = norm(np.cross(up, gaze))   # perpendiular to gaze and up, left-right in camera frame
    V = norm(np.cross(gaze,U))     # top-bottom in camera frame
    W = -norm(gaze)                # camera frame convention, look down "-Z"

    print "U ",U
    print "V ",V
    print "W ",W

    # construct rotation matrix using the normalized basis vectors 
    r = np.identity(4)
    r[:3,0] = U
    r[:3,1] = V
    r[:3,2] = W
    print "r\n",r
   
    # homogenous 4x4 matrix to translate eye to origin
    t = np.identity(4)
    t[:3,3] = -eye
    print "t\n", t

    # translate then rotate, transposed to get inverse
    m = np.dot(r.T,t)  
    print "m\n", m

    return m


def check_world_to_camera( eye, look, up , xeye, xlook, xup ):

    m = world_to_camera(eye, look, up )

    eye_c = np.dot(m,np.append(eye,1))
    look_c = np.dot(m,np.append(look,1))
    up_c = np.dot(m,np.append(up,1))

    print "eye_c", eye_c
    print "look_c", look_c
    print "up_c", up_c

    if not xeye is None: 
        assert np.allclose( xeye ,   eye_c ), (xeye, eye_c)
    if not xlook is None: 
        assert np.allclose( xlook,   look_c ),(xlook, look_c)
    if not xup is None: 
        assert np.allclose( xup,     up_c ), (xup, up_c)


def test_world_to_camera_simplest():
    """
    Simplest expectations for these eye, look, up 
    """
    eye  = np.array([0,0,0])
    look = np.array([0,0,10])
    up = np.array([0,1,0])

    xeye  = np.append(eye,1)
    xlook = np.append(-look,1)
    xup   = np.append(up,1)
   
    check_world_to_camera( eye, look, up, xeye, xlook, xup)

def test_world_to_camera_two():
    """
    """
    eye  = np.array([10,10,10])
    look = np.array([20,20,20])
    up = np.array([0,1,0])

    xeye  = np.append([0,0,0],1)
    xlook = np.append([0,0,-np.linalg.norm(eye-look)],1)
    xup   = None
   
    check_world_to_camera( eye, look, up, xeye, xlook, xup)







if __name__ == '__main__':

    #test_world_to_camera_simplest()
    test_world_to_camera_two()




