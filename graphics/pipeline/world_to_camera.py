#!/usr/bin/env python
"""
"""

import numpy as np

def norm( v ):
    return v/np.linalg.norm(v)

def world_to_camera( eye, look, up, invert=False, debug=False ):
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

    if debug:
        print "eye  %s [%s] " % ( eye , np.linalg.norm(eye))
        print "look %s [%s] " % ( look, np.linalg.norm(look))
        print "gaze %s [%s] " % ( gaze, np.linalg.norm(gaze))
        print "up   %s [%s] " % ( up  , np.linalg.norm(up))

    # UVW basis vectors for the camera frame

    # this had wrong signs for X and Y 
    #U = norm(np.cross(up, gaze))   # perpendiular to gaze and up, left-right in camera frame
    #V = norm(np.cross(gaze,U))     # top-bottom in camera frame
    #W = -norm(gaze)                # camera frame convention, look down "-Z"

    # this way is consistent with ViewTransform
    U = norm(np.cross(gaze,up))   # perpendiular to gaze and up, left-right in camera frame
    V = norm(np.cross(U,gaze))     # top-bottom in camera frame
    W = -norm(gaze)                # camera frame convention, look down "-Z"


    if debug:
        print "U ",U
        print "V ",V
        print "W ",W

    # construct rotation matrix using the normalized basis vectors 
    r = np.identity(4)
    r[:3,0] = U
    r[:3,1] = V
    r[:3,2] = W
  
    # homogenous 4x4 matrix to translate eye to origin
    t = np.identity(4)

    if invert:
        # here for debugging
        t[:3,3] = eye
        m = np.dot(t,r)  
    else:
        # translate then rotate, transposed to get inverse
        t[:3,3] = -eye
        m = np.dot(r.T,t)  
   

    
    if debug:
        print "r\n",r
        print "t\n", t
        print "m\n", m

    return m


def check_world_to_camera( eye, look, up , xeye, xlook, xup, debug=False ):


    gaze = look - eye  
    distance = np.linalg.norm(gaze)
    right = np.cross( gaze, up )

    


    m = world_to_camera(eye, look, up )

    eye_c = np.dot(m,np.append(eye,1))
    look_c = np.dot(m,np.append(look,1))
    up_c = np.dot(m,np.append(up,1))

    if debug: 
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


def test_world_to_camera_random():

    eye = (np.random.random(3) - 0.5)*100
    look = (np.random.random(3) - 0.5)*100
    up = (np.random.random(3) - 0.5)*100

    xeye  = np.append([0,0,0],1)
    xlook = np.append([0,0,-np.linalg.norm(eye-look)],1)
    xup   = None
   
    check_world_to_camera( eye, look, up, xeye, xlook, xup)





if __name__ == '__main__':

    test_world_to_camera_simplest()
    test_world_to_camera_two()
    for _ in range(100):
        test_world_to_camera_random()


    eye = np.array((0,0,0))
    look = np.array((0,0,-10))
    up = np.array((0,1,0))
    gaze = look - eye  

    distance = np.linalg.norm(gaze)
    right = norm( np.cross( gaze, up ))

    m = world_to_camera(eye, look, up )

    # transform points in left-right line of target
    for f in np.linspace(-1,1,9):
        chk = look + right*f  
        chk_c = np.dot(m,np.append(chk,1))
        print f
        print chk
        print chk_c
 




