#!/usr/bin/env python

import logging, time
log = logging.getLogger(__name__)
import numpy as np
import numpy.core.arrayprint as arrayprint
import contextlib

normalize_ = lambda _:_/np.linalg.norm(_)




@contextlib.contextmanager
def printoptions(strip_zeros=True, **kwargs):
    """
    http://stackoverflow.com/questions/2891790/pretty-printing-of-numpy-array
    """
    origcall = arrayprint.FloatFormat.__call__
    def __call__(self, x, strip_zeros=strip_zeros):
        return origcall.__call__(self, x, strip_zeros)
    arrayprint.FloatFormat.__call__ = __call__
    original = np.get_printoptions()
    np.set_printoptions(**kwargs)
    yield 
    np.set_printoptions(**original)
    arrayprint.FloatFormat.__call__ = origcall



def invert_homogenous( m ):
    """
    http://www.euclideanspace.com/maths/geometry/affine/matrix4x4/

    This is not a general inversion, it makes 
    use of the special properties of rotation-translation matrices

    The matrix multiplication order is critical, 
    opposite order works for rotation only but not for rotation + translation matrices
    """

    r = np.identity(4)
    r[:3,:3] = m[:3,:3]     # rotation portion

    t = np.identity(4)
    t[:3,3] = -m[:3,3]      # negate translation portion

    return np.dot(r.T,t)    # transposed rotation * negated translation 



def scale_matrix(scale):
    matrix = np.identity(4)
    matrix[0,0] = scale
    matrix[1,1] = scale
    matrix[2,2] = scale
    return matrix

def translate_matrix(translate):
    matrix = np.identity(4)
    matrix[:3,3] = translate[:3]
    return matrix
    



class Transform(object):
    def __init__(self):
        self.matrix = np.identity(4)
    def __call__(self, v, w=1.):
        return np.dot( self.matrix, np.append(v,w) )
    def __repr__(self):
        with printoptions(precision=3, suppress=True, strip_zeros=False):
             return str(self.matrix)

class ModelToWorld(Transform):
    """
    :param scale:
    :param translate:

    The translation is expected to be "scaled already" 

    """ 
    inverse = False
    def __init__(self, extent, center ): 
        self.extent = extent
        self.center = center

        if self.inverse:
            scale = scale_matrix(1./extent)
            translate = translate_matrix(-center)
            matrix = np.dot( scale, translate )
        else: 
            scale = scale_matrix(extent)
            translate = translate_matrix(center)
            matrix = np.dot( translate, scale )  # equivalent to stuffing in the translation, ie translation does not get scaled

        self.matrix = matrix

class WorldToModel(ModelToWorld):
    inverse = True

class WorldToCamera(Transform):
    def __init__(self, eye, look, up):
         self.matrix = view_transform( eye, look, up, inverse=False )
         
class CameraToWorld(Transform):
    def __init__(self, eye, look, up):
         self.matrix = view_transform( eye, look, up, inverse=True )

 

def view_transform( eye, look, up, inverse=False ):
    """ 
    NB actual view transform in use adopts gluLookAt, this
    is here as a check and in order to obtain the inverse
    of gluLookAt

    OpenGL eye space convention with forward as -Z
    means that have to negate the forward basis vector in order 
    to create a right-handed coordinate system.

    Construct matrix using the normalized basis vectors::    

                             -Z
                       +Y    .  
                        |   .
                  EY    |  .  -EZ forward 
                  top   | .  
                        |. 
                        E-------- +X
                       /  EX right
                      /
                     /
                   +Z

    """
    eye = np.array(eye[:3])
    look = np.array(look[:3])
    up  = np.array(up[:3])

    gaze = look - eye
    distance = np.linalg.norm(gaze)

    #assert len(eye) == 3, eye
    #assert len(look) == 3, look
    #assert len(up) == 3, up
    #assert len(gaze) == 3, gaze

    # orthonormal unit vectors for the camera
    forward = normalize_(gaze)                     # -Z
    #assert len(forward) == 3, forward

    right   = normalize_(np.cross(forward, up))    # +X 
    #assert len(right) == 3, right

    top     = normalize_(np.cross(right,forward)) # +Y 
    #assert len(top) == 3, top
 
    r = np.identity(4)
    r[:3,0] = right      # X column
    r[:3,1] = top        # Y column
    r[:3,2] = -forward   # Z column

    if inverse:
        m = np.dot(translate_matrix(eye),r)
        #
        # camera2world, un-rotate first (eye already at origin)
        # then translate back to world 
        # 
    else:
        m = np.dot(r.T,translate_matrix(-eye))   
        #
        # world2camera, must translate first putting the eye at the origin
        # then rotate to point -Z forward
        #
    return m 





def check_view_transform():
    """ 
    World frame::

             Y
             |
             |       L
             |     /   
             |   E 
             |
             O----------- X


    Camera Frame (O is behind the camera)::

                   -X 
                    |
                    |
            ---O----E-->--L--- -Z
                    |
                    |
                   +X

    """
    eye = (10,10,0)
    look = (20,20,0)
    up = (0,0,1)

    w2c = view_transform( eye, look, up)
    c2w = view_transform( eye, look, up, inverse=True)

    with printoptions(precision=3, suppress=True, strip_zeros=False):
        print w2c
        for p in ([eye,look]):
            world = np.append(p,1)     # homogenize
            camera = np.dot(w2c,world)
            world2 = np.dot(c2w,camera)
            assert np.allclose( world, world2 ) 
            print "world ",world, "camera ", camera, "world2 ",world2


def check_view_transform_2():

    origin = (0,0,0)
    eye = (10,10,0)
    look = (20,20,0)

    up = (0,0,1)

    w2c = WorldToCamera( eye, look, up )
    c2w = CameraToWorld( eye, look, up )
    print "w2c\n", w2c
    print "c2w\n", c2w

    with printoptions(precision=3, suppress=True, strip_zeros=False):
        for world in origin,eye,look:
            camera = w2c(world)[:3]
            world2 = c2w(camera)[:3]
            assert np.allclose( world, world2 ) 
            print "world ",world, "camera ", camera, "world2 ",world2



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    check_view_transform() 
    check_view_transform_2() 
    




