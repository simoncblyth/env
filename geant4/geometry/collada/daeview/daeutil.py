#!/usr/bin/env python
import numpy as np
import numpy.core.arrayprint as arrayprint
import contextlib





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
    matrix[:3,3] = translate 
    return matrix


"""

In [55]: scale_matrix(100)
Out[55]: 
array([[ 100.,    0.,    0.,    0.],
       [   0.,  100.,    0.,    0.],
       [   0.,    0.,  100.,    0.],
       [   0.,    0.,    0.,    1.]])

In [56]: translate_matrix((1,2,3))
Out[56]: 
array([[ 1.,  0.,  0.,  1.],
       [ 0.,  1.,  0.,  2.],
       [ 0.,  0.,  1.,  3.],
       [ 0.,  0.,  0.,  1.]])

In [57]: np.dot(scale_matrix(100),translate_matrix((1,2,3)))
Out[57]: 
array([[ 100.,    0.,    0.,  100.],
       [   0.,  100.,    0.,  200.],
       [   0.,    0.,  100.,  300.],
       [   0.,    0.,    0.,    1.]])

In [58]: np.dot(translate_matrix((1,2,3)),scale_matrix(100))
Out[58]: 
array([[ 100.,    0.,    0.,    1.],
       [   0.,  100.,    0.,    2.],
       [   0.,    0.,  100.,    3.],
       [   0.,    0.,    0.,    1.]])

In [59]: np.dot(scale_matrix(1./100),translate_matrix((-1,-2,-3)))
Out[59]: 
array([[ 0.01,  0.  ,  0.  , -0.01],
       [ 0.  ,  0.01,  0.  , -0.02],
       [ 0.  ,  0.  ,  0.01, -0.03],
       [ 0.  ,  0.  ,  0.  ,  1.  ]])

In [60]: np.dot(np.dot(translate_matrix((1,2,3)),scale_matrix(100)),np.dot(scale_matrix(1./100),translate_matrix((-1,-2,-3))))
Out[60]: 
array([[ 1.,  0.,  0.,  0.],
       [ 0.,  1.,  0.,  0.],
       [ 0.,  0.,  1.,  0.],
       [ 0.,  0.,  0.,  1.]])


"""



class Transform(object):
    def __init__(self):
        self.matrix = np.identity(4)
    def __call__(self, v, w=1.):
        return np.dot( self.matrix, np.append(v,w) )

class ModelToWorldTransform(Transform):
    """
    :param scale:
    :param translate:

    The translation is expected to be "scaled already" 

    """ 
    invert = False
    def __init__(self, extent, center ): 
        self.extent = extent
        self.center = center

        if self.invert:
            scale = scale_matrix(1./extent)
            translate = translate_matrix(-center)
            matrix = np.dot( scale, translate )
        else: 
            scale = scale_matrix(extent)
            translate = translate_matrix(center)
            matrix = np.dot( translate, scale )  # equivalent to stuffing in the translation, ie translation does not get scaled

        self.matrix = matrix

class WorldToModelTransform(ModelToWorldTransform):
    invert = True




