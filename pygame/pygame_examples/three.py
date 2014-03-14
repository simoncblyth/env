#!/usr/bin/env python
"""
The approach taken here for projections got too messy 
and difficult to debug. 
Instead have plumped for full pipeline monty in env.graphics.pipeline.world_to_screen

TODO:

Make this mess go away after salvaged a few things, like Quaternion.


https://pyrr.readthedocs.org/en/latest/
https://pyrr.readthedocs.org/en/latest/_modules/pyrr/matrix44.html#create_perspective_projection_matrix

"""
import math
import numpy as np

class Matrix4(object):
    def __init__(self):
        self.elements = np.identity(4)

    @classmethod
    def fromSymmetricFrustum(cls, left_right, bottom_top, near, far):
        """
        Its easier to determine Frustum parameters 
        based on model vertex extents 
        rather than guess fov etc.. needed for Perspective
        """
        m = cls()
        m.makeFrustum( -left_right, left_right,  -bottom_top, bottom_top, near, far )
        return m 

    @classmethod
    def fromPerspective(cls, yfov, aspect, znear, zfar):
        """

        ::

            yfov 30 znear 1.0 ymin -0.267949192431 ymax 0.267949192431 xmin -0.401923788647 xmax 0.401923788647 
            yfov 30 znear 11.0 ymin -2.94744111674 ymax 2.94744111674 xmin -4.42116167511 xmax 4.42116167511 
            yfov 30 znear 21.0 ymin -5.62693304105 ymax 5.62693304105 xmin -8.44039956158 xmax 8.44039956158 
            yfov 30 znear 31.0 ymin -8.30642496536 ymax 8.30642496536 xmin -12.459637448 xmax 12.459637448 
            yfov 30 znear 41.0 ymin -10.9859168897 ymax 10.9859168897 xmin -16.4788753345 xmax 16.4788753345 
            yfov 30 znear 51.0 ymin -13.665408814 ymax 13.665408814 xmin -20.498113221 xmax 20.498113221 
            yfov 30 znear 61.0 ymin -16.3449007383 ymax 16.3449007383 xmin -24.5173511074 xmax 24.5173511074 
            yfov 30 znear 71.0 ymin -19.0243926626 ymax 19.0243926626 xmin -28.5365889939 xmax 28.5365889939 
            yfov 30 znear 81.0 ymin -21.7038845869 ymax 21.7038845869 xmin -32.5558268804 xmax 32.5558268804 
            yfov 30 znear 91.0 ymin -24.3833765112 ymax 24.3833765112 xmin -36.5750647668 xmax 36.5750647668 

        """
        m = cls()
        m.makePerspective(yfov, aspect, znear, zfar )
        return m 


    def makeFrustum(self, left, right, bottom, top, near, far):
        """
        http://www.opengl.org/sdk/docs/man2/xhtml/glFrustum.xml

        :param left, right: Specify the coordinates for the left and right vertical clipping planes.
        :param bottom, top: Specify the coordinates for the bottom and top horizontal clipping planes.
        :param near, far: Specify the distances to the near and far depth clipping planes. Both distances must be positive.
                 
        #. near must never be set to 0.
        #. near has a scaling effect on output coords

        """
        te = self.elements

        x = 2. * near / ( right - left )
        y = 2. * near / ( top - bottom )
        a = ( right + left ) / ( right - left );
        b = ( top + bottom ) / ( top - bottom );
        c = -1. * ( far + near ) / ( far - near );
        d = -2. * far * near / ( far - near );

        te[0] = [x,0,a,0];
        te[1] = [0,y,b,0];
        te[2] = [0,0,c,d];
        te[3] = [0,0,-1,0];

    def makePerspective(self, yfov, aspect, znear, zfar):
        """
        set up a perspective projection matrix

        http://www.opengl.org/sdk/docs/man2/xhtml/gluPerspective.xml
         
        :param yfov: Specifies the field of view angle, in degrees, in the y direction.
        :param aspect: Specifies the aspect ratio that determines the field of view in the x direction. 
                       The aspect ratio is the ratio of x (width) to y (height). 
                       Aspect 2.0 is twice as wide as tall.
        :param znear: Specifies the distance from the viewer to the near clipping plane (always positive).
        :param zfar: Specifies the distance from the viewer to the far clipping plane (always positive).


        ::

                               |
                               * ymax
                              /|
                             / |
                            /  |
                           /   |
                          /    | 
                         /     | 
                        /      |
                       /       |
                      /        |  
                     /         |
                    /          |
                 | /           | 
                 |/ yfov/2     |
            -----O-------------|--------------------------------------
                 |\ yfov/2   +znear                                +zfar
                 | \           |
                    \          |
                     \         |

        """
        ymax = znear * math.tan( yfov * 0.5 * math.pi / 180. );
        ymin = -ymax;
        xmin = ymin * aspect;
        xmax = ymax * aspect;

        print "yfov %s znear %s ymin %s ymax %s xmin %s xmax %s " % (yfov, znear, ymin, ymax, xmin, xmax )

        self.makeFrustum( xmin, xmax, ymin, ymax, znear, zfar );



class Quaternion(object):
    """
    https://github.com/mrdoob/three.js/blob/master/src/math/Quaternion.js
    """
    @classmethod
    def fromAxisAngle(cls, axis, angle, normalize=False):
        q = cls()
        q.setFromAxisAngle(axis, angle, normalize=normalize)
        return q

    def __init__(self, x = 0, y = 0, z = 0, w = None):
        self.x, self.y, self.z = float(x), float(y), float(z)
        self.w = 1. if w is None else w ;

    def setFromAxisAngle(self, axis, angle, normalize=False):
        """
        http://www.euclideanspace.com/maths/geometry/rotations/conversions/angleToQuaternion/index.htm
        axis have to be normalized
        """
        if normalize:
            axis = axis / np.linalg.norm(axis)

        halfAngle = angle / 2.
        s = math.sin(halfAngle)

        self.x = axis[0] * s
        self.y = axis[1] * s
        self.z = axis[2] * s
        self.w = math.cos(halfAngle)

    def __repr__(self):
        return "%s %s  (%s,%s,%s) " % (self.__class__.__name__, self.w, self.x,self.y,self.z )   



class Point3:
    def __init__(self, x = 0, y = 0, z = 0):
        self.x, self.y, self.z = float(x), float(y), float(z)

    def __repr__(self):
        return "P %s %s %s " % (self.x, self.y, self.z )


    def rotateX(self, angle):
        """ Rotates the point around the X axis by the given angle in radians """
        cosa = math.cos(angle)
        sina = math.sin(angle)
        y = self.y * cosa - self.z * sina
        z = self.y * sina + self.z * cosa
        return Point3(self.x, y, z)
 
    def rotateY(self, angle):
        """ Rotates the point around the Y axis by the given angle in radians. """
        cosa = math.cos(angle)
        sina = math.sin(angle)
        z = self.z * cosa - self.x * sina
        x = self.z * sina + self.x * cosa
        return Point3(x, self.y, z)
 
    def rotateZ(self, angle):
        """ Rotates the point around the Z axis by the given angle in radians. """
        cosa = math.cos(angle)
        sina = math.sin(angle)
        x = self.x * cosa - self.y * sina
        y = self.x * sina + self.y * cosa
        return Point3(x, y, self.z)

 
    def project(self, width, height, fov, viewer_distance):
        """ 
        :param width:  window width in pixels
        :param height:  window height in pixels
        :param fov:  not field-of-view
        :param viewer_distance: specifies viewpoint position at (0,0,viewer_distance) 

        Transforms this 3D point to 2D using a perspective projection. 
        """
        factor = fov / (viewer_distance + self.z)
        x = self.x * factor + width / 2
        y = self.y * factor + height / 2
        return Point3(x, y, self.z)


    def copy(self):
        return Point3(self.x, self.y, self.z)
 
    def applyQuaternion(self, q ):
        """
        https://github.com/mrdoob/three.js/blob/master/src/math/Vector3.js
        """
        x = self.x;
        y = self.y;
        z = self.z;

        qx = q.x;
        qy = q.y;
        qz = q.z;
        qw = q.w;

        # calculate quat * vector

        ix =  qw * x + qy * z - qz * y;
        iy =  qw * y + qz * x - qx * z;
        iz =  qw * z + qx * y - qy * x;
        iw = -qx * x - qy * y - qz * z;

        # calculate result * inverse quat

        self.x = ix * qw + iw * -qx + iy * -qz - iz * -qy;
        self.y = iy * qw + iw * -qy + iz * -qx - ix * -qz;
        self.z = iz * qw + iw * -qz + ix * -qy - iy * -qx;


    def applyProjection(self, m ):
        """
        https://github.com/mrdoob/three.js/blob/master/src/math/Vector3.js
        """
        x = self.x;
        y = self.y;
        z = self.z;
 
 
        # **F**ortran (column-major) order, rather than default **C** (row-major) order
        e = m.elements.flatten('F');  
        den = ( e[3] * x + e[7] * y + e[11] * z + e[15] ); 

        print "e3 %s x %s e7 %s y %s e11 %s z %s e15 %s " % (e[3],x,e[7],y,e[11],z,e[15])


        #print x,y,x
        #print e
        #print den

        if abs(den) < 0.0001:
           print "small den %s " % den 
           den = 0.0001
 
        d = 1. / den ; # perspective divide

        self.x = ( e[0] * x + e[4] * y + e[8]  * z + e[12] ) * d;
        self.y = ( e[1] * x + e[5] * y + e[9]  * z + e[13] ) * d;
        self.z = ( e[2] * x + e[6] * y + e[10] * z + e[14] ) * d;



if __name__ == '__main__':

    #for znear in range(1,100,10):
    #    m = Matrix4.fromPerspective(30, 1.5, float(znear), float(znear)*10. )
    #    #print m.elements
    

    znear = 5.
    m = Matrix4.fromPerspective(30., 1.5, float(znear), float(znear)*10. )


    points = [
           ('O.',Point3(0,0,0),),
           ('+X',Point3(1,0,0),),
           ('+Y',Point3(0,1,0),),
           ('+Z',Point3(0,0,1),),
           ('-X',Point3(-1,0,0),),
           ('-Y',Point3(0,-1,0),),
           ('-Z',Point3(0,0,-1),),
           ('-tX',Point3(-10,0,0),),
           ('-tY',Point3(0,-10,0),),
           ('-tZ',Point3(0,0,-10),),
         ]
 
    for label, point in points:
        t = point.copy()
        t.applyProjection(m)
        print " %s  %s    %s " % ( label, point, t  )




 


