OpenGL Transformations
========================


lookAt
--------


gluLookAt
-----------

* http://www.opengl.org/archives/resources/faq/technical/viewing.htm

gluLookAt() takes an eye position, a position to look at, and an up vector, 
all in object space coordinates and computes the inverse camera transform according to
its parameters and multiplies it onto the current matrix stack.

The GL_PROJECTION matrix should contain only the projection transformation
calls it needs to transform eye space coordinates into clip coordinates.

The GL_MODELVIEW matrix, as its name implies, should contain modeling and
viewing transformations, which transform object space coordinates into eye
space coordinates. Remember to place the camera transformations on the
GL_MODELVIEW matrix and never on the GL_PROJECTION matrix.

Think of the projection matrix as describing the attributes of your camera,
such as field of view, focal length, fish eye lens, etc. Think of the ModelView
matrix as where you stand with the camera and the direction you point it.


reminders on matrix non-commutative properties
------------------------------------------------

::

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






