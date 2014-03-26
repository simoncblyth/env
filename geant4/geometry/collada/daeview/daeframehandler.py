#!/usr/bin/env python
"""


"""

import OpenGL.GL as gl
import OpenGL.GLU as glu


def gl_modelview_matrix():
    return gl.glGetDoublev( gl.GL_MODELVIEW_MATRIX )

def gl_projection_matrix():
    return gl.glGetDoublev( gl.GL_PROJECTION_MATRIX )


class DAEFrameHandler(object):
    def __init__(self, frame, mesh, scene, config ):
        self.frame = frame
        self.mesh = mesh
        self.scene = scene
        self.trackball = scene.trackball
        pass
        args = config.args
        self.fill = args.fill
        self.line = args.line
        self.transparent = args.transparent
        self.light = args.light
        pass
        frame.push(self) # declares this object to handle event notifications from the frame

    # toggles invoked by Interactivity Handler
    def toggle_fill(self):
        self.fill = not self.fill
    def toggle_line(self):
        self.line = not self.line
    def toggle_transparent(self):
        self.transparent = not self.transparent
    def toggle_parallel(self):
        self.trackball.parallel = not self.trackball.parallel


    def on_init(self):
        """
        Enabling lights 0,1,2 gives acceptable lighting 
        with a white background. Enabling them singly 
        gives nasty red/green/blue lighting and background.

        glumpy.Figure.on_init sets up the RGB lights before dispatch
        to all figures

        http://www.opengl.org/archives/resources/faq/technical/lights.htm

        """
        print "FrameHandler on_init"
        if self.light:
            refextent = self.trackball.refextent
            print "FrameHandler enabling lights %s " % refextent
            self.lights(refextent, w=1.0)
        pass

    def lights(self, d, w=0.):
        """
        With w=0. get at very colorful render,  that changes 
        to different colors as rotate around.

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
        gl.glLightfv (gl.GL_LIGHT0, gl.GL_POSITION,( -d,   d,   d,   w))
        gl.glLightfv (gl.GL_LIGHT1, gl.GL_POSITION,(  d,   d,   d,   w))
        gl.glLightfv (gl.GL_LIGHT2, gl.GL_POSITION,(0.0,  -d,   d,   w))

    def gluLookAt(self, trackball):
        """ 
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
        eye, look, up = trackball.eye, trackball.look, trackball.up
        glu.gluLookAt(eye[0],eye[1],eye[2],look[0],look[1],look[2],up[0],up[1],up[2])

    def push(self, trackball):
        gl.glMatrixMode (gl.GL_MODELVIEW)
        gl.glPushMatrix()
        gl.glLoadIdentity ()

        refextent = trackball.refextent
        if trackball.lookat:
            gl.glTranslate ( trackball._x*refextent, trackball._y*refextent, -trackball._z*refextent )
            gl.glMultMatrixf (trackball._matrix)
            self.gluLookAt(trackball)        # puts the camera at origin looking down Z
        else:
            gl.glTranslate (trackball._x, trackball._y, -trackball._z )
            gl.glMultMatrixf (trackball._matrix)
        pass

        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glPushMatrix()
        gl.glLoadIdentity ()

        lrbtnf = trackball.lrbtnf 
        #print "lrbtnf", lrbtnf
        if trackball.parallel:
            gl.glOrtho ( *lrbtnf )
        else:
            gl.glFrustum ( *lrbtnf )
        pass
        # refextent scale down to bring near to -1:1 range, in non-lookat mode refextent is 1.0 anyhow 
        if trackball.lookat:
            gl.glScalef(1./refextent, 1./refextent, 1./refextent)  

    def pop(self, trackball):
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glPopMatrix()
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glPopMatrix()


    def on_draw(self):
        self.frame.lock()
        self.frame.draw()

        self.push(self.trackball)  # sets up the matrices

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

        self.pop(self.trackball)
        self.frame.unlock()




if __name__ == '__main__':
    pass



