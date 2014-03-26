#!/usr/bin/env python
"""


Division of concerns

`DAEFrameHandler`
         visual presentation
`DAETrackball`
         rotation and projection
`DAEView`
         position


lights
------

* http://www.opengl.org/archives/resources/faq/technical/lights.htm

* http://www.talisman.org/opengl-1.1/Reference/glLight.html

The position is transformed by the modelview matrix when glLight is called
(just as if it were a point), and it is stored in eye coordinates. If the w
component of the position is 0, the light is treated as a directional source.
Diffuse and specular lighting calculations take the light's direction, but not
its actual position, into account, and attenuation is disabled. Otherwise,
diffuse and specular lighting calculations are based on the actual location of
the light in eye coordinates, and attenuation is enabled. The initial position
is (0, 0, 1, 0); thus, the initial light source is directional, parallel to,
and in the direction of the -Z axis.


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


"""
import math
import numpy as np
import OpenGL.GL as gl
import OpenGL.GLU as glu

def gl_modelview_matrix():
    return gl.glGetDoublev( gl.GL_MODELVIEW_MATRIX )

def gl_projection_matrix():
    return gl.glGetDoublev( gl.GL_PROJECTION_MATRIX )

def oscillate( count, low, high, speed ):
    return low + (high-low)*(math.sin(count*math.pi*speed)+1.)/2.



count = 0 

class DAEFrameHandler(object):
    """
    Handles event notifications from the frame: `on_init` and `on_draw`
    """
    def __init__(self, frame, mesh, scene, config ):
        self.frame = frame
        self.mesh = mesh
        pass
        self.scene = scene
        self.trackball = scene.trackball
        self.view = scene.view
        
        lookat = not scene.scaled_mode
        if lookat:
            scale = scene.extent
        else:
            scale = 1.

        self.lookat = lookat 
        self.scale = scale
        self.tweak = 1.

        pass
        args = config.args
        self.fill = args.fill
        self.line = args.line
        self.transparent = args.transparent
        self.light = args.light
        self.parallel = args.parallel
        self.animate = False
        self.speed = args.speed
        pass
        frame.push(self) 

    def __repr__(self):
        return "scale %7.2f tweak %7.2f " %  (self.scale, self.tweak )

    # toggles invoked by Interactivity Handler
    def toggle_fill(self):
        self.fill = not self.fill
    def toggle_line(self):
        self.line = not self.line
    def toggle_transparent(self):
        self.transparent = not self.transparent
    def toggle_parallel(self):
        self.parallel = not self.parallel
    def toggle_animate(self):
        self.animate = not self.animate
    def tweak_scale(self, factor ):
        self.tweak *= factor

    def tick(self, dt):
        """
        invoked from Interactivity handlers on_idle as this is not getting those notifications

        hmm better way to prevent this being called too often ?
        """
        if not self.animate:return
        global count
        count += 1  
        self.view(count, self.speed)
        self.frame.redraw() 

    def on_init(self):
        """
        glumpy.Figure.on_init sets up the RGB lights before dispatch to all figures
        """
        if self.light:
            self.lights(self.scale, w=1.0)
        pass

    def lights(self, d, w=0.):
        """
        With w=0. get at very colorful render,  that changes 
        to different colors as rotate around.

        scale is not enough, need to transform the positions
        """
        gl.glLightfv (gl.GL_LIGHT0, gl.GL_DIFFUSE, (1.0, 0.0, 0.0, 1.0))
        gl.glLightfv (gl.GL_LIGHT0, gl.GL_AMBIENT, (0.0, 0.0, 0.0, 1.0))
        gl.glLightfv (gl.GL_LIGHT0, gl.GL_SPECULAR,(0.0, 0.0, 0.0, 0.0))
        gl.glLightfv (gl.GL_LIGHT0, gl.GL_POSITION,( -d,   d,   d,   w))

        gl.glLightfv (gl.GL_LIGHT1, gl.GL_DIFFUSE, (0.0, 1.0, 0.0, 1.0))
        gl.glLightfv (gl.GL_LIGHT1, gl.GL_AMBIENT, (0.0, 0.0, 0.0, 0.0))
        gl.glLightfv (gl.GL_LIGHT1, gl.GL_SPECULAR,(0.0, 0.0, 0.0, 0.0))
        gl.glLightfv (gl.GL_LIGHT1, gl.GL_POSITION,(  d,   d,   d,   w))

        gl.glLightfv (gl.GL_LIGHT2, gl.GL_DIFFUSE, (0.0, 0.0, 1.0, 1.0))
        gl.glLightfv (gl.GL_LIGHT2, gl.GL_AMBIENT, (0.0, 0.0, 0.0, 0.0))
        gl.glLightfv (gl.GL_LIGHT2, gl.GL_SPECULAR,(0.0, 0.0, 0.0, 0.0))
        gl.glLightfv (gl.GL_LIGHT2, gl.GL_POSITION,(0.0,  -d,   d,   w))

        gl.glEnable (gl.GL_LIGHTING) # with just this line tis very dark, but no nasty red
        gl.glEnable (gl.GL_LIGHT0)  
        gl.glEnable (gl.GL_LIGHT1)  
        gl.glEnable (gl.GL_LIGHT2)   

    def light_position(self, d, w=0.):
        """
        Hmm to place the lights in world frame, need to rustle up appropriate 
        coordinates equivalent to the old scaled regime. Cannot just set d and
        hope for the best that just corresponds to a scaling, there is offset too.
        """
        gl.glLightfv (gl.GL_LIGHT0, gl.GL_POSITION,( -d,   d,   d,   w))
        gl.glLightfv (gl.GL_LIGHT1, gl.GL_POSITION,(  d,   d,   d,   w))
        gl.glLightfv (gl.GL_LIGHT2, gl.GL_POSITION,(0.0,  -d,   d,   w))

    def push(self):

        scene = self.scene
        trackball = self.trackball
        view = self.view
        scale = self.scale
        factor = scale*self.tweak

        gl.glMatrixMode (gl.GL_MODELVIEW)
        gl.glPushMatrix()
        gl.glLoadIdentity ()

        gl.glTranslate ( trackball._x*factor, trackball._y*factor, -trackball._z*factor )
        gl.glMultMatrixf (trackball._matrix )   # rotation only 

        if self.lookat:
            glu.gluLookAt( *view.eye_look_up )
        else:
            pass   # positioned via vbo scaling and centering  


        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glPushMatrix()
        gl.glLoadIdentity ()

        lrbtnf = trackball.lrbtnf * factor
        if self.parallel:
            gl.glOrtho ( *lrbtnf )
        else:
            gl.glFrustum ( *lrbtnf )
        pass

        gl.glScalef(1./scale, 1./scale, 1./scale)   # does nothing for scaled mode


    def pop(self):
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glPopMatrix()
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glPopMatrix()


    def on_draw(self):
        self.frame.lock()
        self.frame.draw()

        self.push() # matrices

        #if self.light:
        #    self.light_position( 0.9, 1.)   # reset positions following changes to MODELVIEW matrix ?

        if self.fill:
            if self.transparent:
                gl.glEnable (gl.GL_BLEND)
                gl.glBlendFunc ( gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
            pass
            gl.glEnable (gl.GL_COLOR_MATERIAL)
            gl.glColorMaterial(gl.GL_FRONT_AND_BACK, gl.GL_AMBIENT_AND_DIFFUSE)

            gl.glEnable( gl.GL_POLYGON_OFFSET_FILL )
            gl.glPolygonOffset (1, 1)
            gl.glPolygonMode( gl.GL_FRONT_AND_BACK, gl.GL_FILL )

            self.mesh.draw( gl.GL_TRIANGLES, "pnc" )   # position,normal,color

            gl.glDisable( gl.GL_POLYGON_OFFSET_FILL )

            if self.transparent:
                gl.glDisable( gl.GL_BLEND )
            pass
        pass

        if self.line:
            gl.glPolygonMode( gl.GL_FRONT_AND_BACK, gl.GL_LINE )
            gl.glEnable( gl.GL_BLEND )
            gl.glEnable( gl.GL_LINE_SMOOTH )
            gl.glColor( 0., 0., 0., 0.5 ) # difficult to see lines from some directions, unless black

            self.mesh.draw( gl.GL_TRIANGLES, "p" )  # position

            gl.glDisable( gl.GL_BLEND )
            gl.glDisable( gl.GL_LINE_SMOOTH )

        self.pop() # matrices
        self.frame.unlock()




if __name__ == '__main__':
    pass



