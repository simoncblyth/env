#!/usr/bin/env python
"""
OPENGL CLIPPING PLANES
========================

OpenGL
-------

* https://www.opengl.org/sdk/docs/man2/xhtml/glClipPlane.xml
* http://pyopengl.sourceforge.net/documentation/manual-3.0/glClipPlane.html

When glClipPlane is called, equation is transformed by the inverse of the
modelview matrix and stored in the resulting eye coordinates. Subsequent
changes to the modelview matrix have no effect on the stored plane-equation
components. If the dot product of the eye coordinates of a vertex with the
stored plane equation components is positive or zero, the vertex is **in** with
respect to that clipping plane. Otherwise, it is **out**.

All clipping planes are initially defined as (0, 0, 0, 0) in eye coordinates and are disabled.

Maths 
------

* http://www.songho.ca/math/plane/plane.html

The equation of a plane is defined with a normal vector
(perpendicular to the plane) and a known point on the plane.::

   ax + by + cz + d = 0

The normal direction gives coefficients (a,b,c) and the single point (x1,y1,z1) 
fixes the plane along that direction via::

   d = -np.dot( [a,b,c], [x1,y1,z1] )  


Implement from daetransform
---------------------------------

To obtain those from current view transforms use:

#. gaze direction `look - eye` (0,0,-1)  [in eye frame] 
#. some convenient point, maybe near point (0,0,-near) or look point (0,0,-distance) [in eye frame] 

Actually more convenient to use the current near clipping plain, as 
can interactively vary that and when have it where desired freeze it
into a fixed clipping plane.

"""

import logging
log = logging.getLogger(__name__)

import OpenGL.GL as gl
import OpenGL.GLUT as glut

PLANE=(gl.GL_CLIP_PLANE0,
       gl.GL_CLIP_PLANE1,
       gl.GL_CLIP_PLANE2,
       gl.GL_CLIP_PLANE3,
       gl.GL_CLIP_PLANE4,
       gl.GL_CLIP_PLANE5)

class DAEClipper(object):
    def __init__(self):
        self.clear()

    def clear(self):
        self.nplane = 0
        self.planes = {}

    def add(self, plane_eqn ):
        """
        :param 4-element array representing plane equation 
        """
        assert len(plane_eqn) == 4, "expecting 4-element array/tuple " 
        if self.nplane > len(PLANE):
            log.warn("cannot add clipping plane %s already used all available planes " % self.nplane )
            return 
        pass
        log.info("add nplane %s plane_eqn %s " % (self.nplane,repr(plane_eqn)))
        self.planes[self.nplane] = plane_eqn 
        self.nplane += 1

    def enable_one(self, n):
        """
        Seems that need to keep defining the clip plane, for every draw ?
        """
        if not n in self.planes:return
        gl.glClipPlane(PLANE[n], self.planes[n] ) 
        gl.glEnable(PLANE[n])

    def disable_one(self, n):
        if not n in self.planes:return
        gl.glDisable(PLANE[n])

    def draw(self):
        self.enable()

    def reset(self):
        self.disable()
        self.clear()

    def enable(self):
        for n in range(self.nplane):
            self.enable_one(n)
 
    def disable(self):
        for n in range(self.nplane):
            self.disable_one(n)
 

if __name__ == '__main__':
    pass


