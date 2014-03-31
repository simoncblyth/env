#!/usr/bin/env python
"""
"""

import numpy as np
import OpenGL.GL as gl
import OpenGL.GLUT as glut


fvec_ = lambda _:map(float, _.split(","))


class DAELights(object):
    """
    Glumpy original light positioning with lights in Z=1 plane::

       (-1,1,1)             (1,1,1)
            R      |      G 
                   |
                   |
            -------+-------
                   |
                   |
                   |    
   
                   B
                (0,-1,1)


    """
    def __init__(self, transform,  config ):
         
        args = config.args
        light0 = fvec_(args.rlight)
        light1 = fvec_(args.glight)
        light2 = fvec_(args.blight)
        wlight = args.wlight
        lights = args.lights
        factor = args.flight

        self.transform = transform
        self._light0 = np.array(light0)*factor
        self._light1 = np.array(light1)*factor
        self._light2 = np.array(light2)*factor
        self.lights = lights 
        self.light  = args.light 
        self.wlight = args.wlight 

    light0 = property(lambda self:self.transform(self._light0,w=self.wlight))
    light1 = property(lambda self:self.transform(self._light1,w=self.wlight))
    light2 = property(lambda self:self.transform(self._light2,w=self.wlight))
   
    def __repr__(self):
        return "\n".join([
                   "%s" % self.__class__.__name__ ,
                   "light0 %s %s " % (str(self.light0),str(self._light0)),
                   "light1 %s %s " % (str(self.light1),str(self._light1)),
                   "light2 %s %s " % (str(self.light2),str(self._light2)),
                       ])
 
    def setup(self):
        """
        With w=0. get at very colorful render,  that changes 
        to different colors as rotate around.

        scale is not enough, need to transform the positions
        """

        #if "r" in self.lights:
        if 1:
            gl.glLightfv (gl.GL_LIGHT0, gl.GL_DIFFUSE, (1.0, 1.0, 1.0, 1.0))
            gl.glLightfv (gl.GL_LIGHT0, gl.GL_AMBIENT, (0.0, 0.0, 0.0, 1.0))
            gl.glLightfv (gl.GL_LIGHT0, gl.GL_SPECULAR,(0.0, 0.0, 0.0, 0.0))
            gl.glLightfv (gl.GL_LIGHT0, gl.GL_POSITION, self.light0 )
            #gl.glLightf(  gl.GL_LIGHT0, gl.GL_CONSTANT_ATTENUATION, .2)


        #if "g" in self.lights:
        if 1:
            gl.glLightfv (gl.GL_LIGHT1, gl.GL_DIFFUSE, (1.0, 1.0, 1.0, 1.0))
            gl.glLightfv (gl.GL_LIGHT1, gl.GL_AMBIENT, (0.0, 0.0, 0.0, 0.0))
            gl.glLightfv (gl.GL_LIGHT1, gl.GL_SPECULAR,(0.0, 0.0, 0.0, 0.0))
            gl.glLightfv (gl.GL_LIGHT1, gl.GL_POSITION, self.light1 )
            #gl.glLightf(  gl.GL_LIGHT1, gl.GL_CONSTANT_ATTENUATION, .2)

        #if "b" in self.lights:
        if 1:
            gl.glLightfv (gl.GL_LIGHT2, gl.GL_DIFFUSE, (1.0, 1.0, 1.0, 1.0))
            gl.glLightfv (gl.GL_LIGHT2, gl.GL_AMBIENT, (0.0, 0.0, 0.0, 0.0))
            gl.glLightfv (gl.GL_LIGHT2, gl.GL_SPECULAR,(0.0, 0.0, 0.0, 0.0))
            gl.glLightfv (gl.GL_LIGHT2, gl.GL_POSITION, self.light2 )
            #gl.glLightf(  gl.GL_LIGHT2, gl.GL_CONSTANT_ATTENUATION, .2)


        if len(self.lights) > 0:
            gl.glEnable (gl.GL_LIGHTING) # with just this line tis very dark
        
        if "r" in self.lights:
            gl.glEnable (gl.GL_LIGHT0)  

        if "g" in self.lights:
            gl.glEnable (gl.GL_LIGHT1)  

        if "b" in self.lights:
            gl.glEnable (gl.GL_LIGHT2)   


    def position(self):
        """
        Hmm to place the lights in world frame, need to rustle up appropriate 
        coordinates equivalent to the old scaled regime. Cannot just set d and
        hope for the best that just corresponds to a scaling, there is offset too.
        """
        if "r" in self.lights:
            gl.glLightfv (gl.GL_LIGHT0, gl.GL_POSITION, self.light0 )
        if "g" in self.lights:
            gl.glLightfv (gl.GL_LIGHT1, gl.GL_POSITION, self.light1 )
        if "b" in self.lights:
            gl.glLightfv (gl.GL_LIGHT2, gl.GL_POSITION, self.light2 )


    def draw(self, distance):

        gl.glDisable( gl.GL_LIGHTING )
        gl.glDisable( gl.GL_DEPTH_TEST )

        gl.glColor3f( 1.,0.,0. )
        gl.glPushMatrix()
        gl.glTranslate ( *self.light0[:3] )
        glut.glutSolidCube( distance/10. )
        gl.glPopMatrix()

        gl.glColor3f( 0.,1.,0. )
        gl.glPushMatrix()
        gl.glTranslate ( *self.light1[:3] )
        glut.glutSolidCube( distance/10. )
        gl.glPopMatrix()

        gl.glColor3f( 0.,0.,1. )
        gl.glPushMatrix()
        gl.glTranslate ( *self.light2[:3] )
        glut.glutSolidCube( distance/10. )
        gl.glPopMatrix()

        gl.glEnable( gl.GL_LIGHTING )
        gl.glEnable( gl.GL_DEPTH_TEST )

        


if __name__ == '__main__':
    pass
