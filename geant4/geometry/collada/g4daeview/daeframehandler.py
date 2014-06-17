#!/usr/bin/env python
"""
"""
import logging
log = logging.getLogger(__name__)
import math
import numpy as np
import OpenGL.GL as gl
import OpenGL.GLU as glu
import OpenGL.GLUT as glut
from OpenGL.GL import GLfloat


from daetext import DAEText
from daeillustrate import DAEIllustrate


def gl_modelview_matrix():
    return gl.glGetDoublev( gl.GL_MODELVIEW_MATRIX )

def gl_projection_matrix():
    return gl.glGetDoublev( gl.GL_PROJECTION_MATRIX )

def oscillate( count, low, high, speed ):
    return low + (high-low)*(math.sin(count*math.pi*speed)+1.)/2.



class DAEFrameHandler(object):
    """
    Keep this for handling graphics presentation, **NOT** interactivity, **NOT state**     

    Handles event notifications from the frame: `on_init` and `on_draw`
    """
    def __init__(self, frame, mesh, scene ):
        self.frame = frame
        self.mesh = mesh
        self.scene = scene
        pass
        self.count = 0 
        self.fps = 0 
        self.timebase = 0 
        pass
        self.text = DAEText()
        self.illustrate = DAEIllustrate()
        self.annotate = []
        pass
        frame.push(self)  # get frame to invoke on_init and on_draw handlers

    def __repr__(self):
        return "FH %s " % self.fps

    def glinfo(self):
        info = {}
        for k in (gl.GL_VERSION, gl.GL_SHADING_LANGUAGE_VERSION, gl.GL_EXTENSIONS):
            info[k.name] = gl.glGetString(k)  
            #log.info("%s %s " % (k, "\n".join(info[k.name].split())  ))

        for k in (gl.GL_MAX_VERTEX_ATTRIBS,):
            info[k.name] = gl.glGetIntegerv(k)  
            log.info("%s %s " % (k, info[k.name]) )

        return info

    def tick(self, dt):
        """
        invoked from Interactivity handlers on_idle as this is not getting those notifications

        hmm better way to prevent this being called too often ?
        """
        if not self.scene.animate:return
        self.scene.tick(dt)
        self.frame.redraw() 

    def on_init(self):
        """
        glumpy.Figure.on_init sets up the RGB lights before dispatch to all figures
        """
        if self.scene.light:
            self.scene.lights.setup()
            log.debug("on_init lights\n%s" % str(self.scene.lights))
        pass


    def push(self):
        """
        Transformations are applied in the reverse order
        to how they appear in OpenGL code.

        Light positions are set using world coordinates but OpenGL
        stores them in eye coordinates by transforming using the MODELVIEW matrix
        in force at the time of the light positioning call.

        This means that for fixed light positions have to keep re-positioning 
        the lights as the MODELVIEW matrix is changed by moving around.

        Thus it matters whether scaling is done in MODELVIEW as 
        opposed to PROJECTION 


        eye/look/up in world frame define the camera transform, 
        translating eye to origin and rotating look to be along -z

        """
        scene = self.scene
        trackball = self.scene.trackball
        camera = self.scene.camera
        view = self.scene.view
        kscale = camera.kscale
        distance = view.distance


        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glPushMatrix()
        gl.glLoadIdentity ()

        if self.scene.parallel:    # parallel should be property of camera, not scene ?
            gl.glOrtho ( *camera.lrbtnf )
        else:
            gl.glFrustum ( *camera.lrbtnf )
        pass


        gl.glMatrixMode (gl.GL_MODELVIEW)
        gl.glPushMatrix()
        gl.glLoadIdentity ()

        gl.glScalef(1./kscale, 1./kscale, 1./kscale)   

        gl.glTranslate ( *trackball.xyz )  # former adhoc 1000. now done internally in trackball.translatefactor

        # temporarily shunt origin to the "look" rather than the "eye", for applying rotation and markers
        gl.glTranslate ( 0, 0, -distance )                         
        gl.glMultMatrixf( (GLfloat*16)(*trackball._matrix) )

        if self.scene.markers:
            glut.glutWireSphere( kscale*trackball.trackballradius,10,10)  # what size trackball ?

        gl.glTranslate ( 0, 0, +distance )                           # look is at (0,0,-distance) in eye frame, so here we shunt to the look

        if not scene.scaled_mode:
            glu.gluLookAt( *view.eye_look_up )   # NB no scaling, still world distances, eye at origin and point -Z at look



    def pop(self):
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glPopMatrix()
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glPopMatrix()


    def on_draw(self):

        self.count += 1
        #time = glut.glutGet( glut.GLUT_ELAPSED_TIME )
        #if (time - self.timebase > 1000):
        #    fps = self.count*1000.0/(time-self.timebase)
        #    pass
        #    #log.info("fps %s " % fps )
        #    self.fps = fps
        #    self.timebase = time;    
        #    self.count = 0;
        #pass


        self.frame.lock()
        self.frame.draw()

        self.push() # matrices

        self.annotate = []

        view = self.scene.view
        lights = self.scene.lights
        distance = self.scene.view.distance
        camera = self.scene.camera
        transform = self.scene.transform
        lrbtnf = camera.lrbtnf
        kscale = camera.kscale          

        self.scene.clipper.draw()

        if self.scene.markers:
            self.illustrate.frustum( view, lrbtnf*kscale )
            self.illustrate.raycast( transform.pixel2world_notrackball , view.eye, camera ) 

        if self.scene.event:
            self.scene.event.draw() 

        if self.scene.light:
            lights.position()   # reset positions following changes to MODELVIEW matrix ?

        if self.scene.markers:
            lights.draw(distance) 

        if len(self.scene.solids)>0:
            for solid in self.scene.solids:
                self.annotate.append(repr(solid))
            pass

        if self.scene.drawsolid:
            if len(self.scene.solids) > 0:
                solid = self.scene.solids[0]
                gl.glColor3f( 1.,0.,0. )
                gl.glPushMatrix()
                gl.glTranslate ( *solid.center )
                glut.glutWireSphere( solid.extent*1.2 , 10, 10)
                gl.glPopMatrix()


        if self.scene.fill:
            if self.scene.transparent:
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

            if self.scene.transparent:
                gl.glDisable( gl.GL_BLEND )
            pass
        pass

        if self.scene.line:
            gl.glPolygonMode( gl.GL_FRONT_AND_BACK, gl.GL_LINE )
            gl.glEnable( gl.GL_BLEND )
            gl.glEnable( gl.GL_LINE_SMOOTH )
            gl.glColor( 0., 0., 0., 0.5 ) # difficult to see lines from some directions, unless black

            self.mesh.draw( gl.GL_TRIANGLES, "p" )  # position

            gl.glDisable( gl.GL_BLEND )
            gl.glDisable( gl.GL_LINE_SMOOTH )


        if self.scene.cuda:
            if self.scene.processor is not None:
                self.scene.processor.process() 
                self.scene.processor.display() 

        if self.scene.raycast:
            self.scene.raycaster.render() 
            self.fig_handler.retitle()   # for raycast times to appear in title immediately 

        self.pop() # matrices

        if len(self.annotate) > 0:
            self.text(self.annotate)

        self.frame.unlock()


    def unproject(self, x, y ):
        """
        Obtain world space 3D coordinate from a mouse/pad click  

        :param x: screen coordinates 
        :param y:

        TODO: check bit depth 
        """
        self.push()

        pixels = gl.glReadPixelsf(x, y, 1, 1, gl.GL_DEPTH_COMPONENT ) # width,height 1,1  
        z = pixels[0][0]
        click_xyz = None 
        try:
            click_xyz = glu.gluUnProject( x,y,z ) # click point in world frame       
        except ValueError:
            log.warn("gluUnProject FAILED for x,y,z %s %s %s" % (x,y,z)) 
        pass

        self.pop()
        return click_xyz


    def write_to_file(self, name, x,y,w,h):
        self.push()

        #format_, type_, pil_format = gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, 'RGBA'
        format_, type_, pil_format = gl.GL_RGB, gl.GL_UNSIGNED_BYTE, 'RGB'

        data = gl.glReadPixels (x,y,w,h, format_, type_)
        from PIL import Image
        image = Image.fromstring( pil_format , (w,h), data)
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        image.save ("%s.png" % name)

        self.pop()



if __name__ == '__main__':
    pass



