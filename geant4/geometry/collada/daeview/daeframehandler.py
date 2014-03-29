#!/usr/bin/env python
"""


Division of concerns

`DAEFrameHandler`
         visual presentation
`DAETrackball`
         rotation and projection
`DAEView`
         position


"""
import logging
log = logging.getLogger(__name__)
import math
import numpy as np
import OpenGL.GL as gl
import OpenGL.GLU as glu
import OpenGL.GLUT as glut


from daeutil import Transform
from daelights import DAELights



class DAEText(object):
    fonts = { 
                   "8_BY_13":dict(code=glut.GLUT_BITMAP_8_BY_13,leading=50.), 
                   "9_BY_15":dict(code=glut.GLUT_BITMAP_9_BY_15), 
            "TIMES_ROMAN_10":dict(code=glut.GLUT_BITMAP_TIMES_ROMAN_10), 
            "TIMES_ROMAN_24":dict(code=glut.GLUT_BITMAP_TIMES_ROMAN_24), 
              "HELVETICA_10":dict(code=glut.GLUT_BITMAP_HELVETICA_10,leading=15.),
              "HELVETICA_12":dict(code=glut.GLUT_BITMAP_HELVETICA_12),
             }

    def __init__(self, font="HELVETICA_10"):
        font = self.fonts[font]
        self.font = font['code']
        self.leading = font.get('leading',20.)

    def check(self):
        self( self.fonts.keys() )

    def __call__(self, lines ):
        gl.glDisable( gl.GL_LIGHTING )
        gl.glDisable( gl.GL_DEPTH_TEST )

        gl.glColor3f( 0.1, 0.1, 0.1 )
        for i, line in enumerate(lines):
            ypos = self.leading * (i+1)
            gl.glRasterPos3f( 2.*self.leading , ypos , 0 )
            for c in line:
                glut.glutBitmapCharacter( self.font, ord(c) )
            pass

        gl.glEnable( gl.GL_LIGHTING )
        gl.glEnable( gl.GL_DEPTH_TEST )




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
        self.scene = scene
        
        lookat = not scene.scaled_mode
        if lookat:
            scale = scene.view.extent
            light_transform = scene.geometry.mesh.model2world 
            log.info("mesh\n%s" % scene.geometry.mesh.smry() )
        else:
            scale = 1.
            light_transform = Transform()
        pass
        self.scale = scale
        self.lights = DAELights( light_transform, config )
        self.lookat = lookat 
        self.settings(config.args)
        pass
        self.text = DAEText()
        self.annotate = []
        pass
        frame.push(self)  # get frame to invoke on_init and on_draw handlers

    def __repr__(self):
        return "F %7.2f " %  (self.scale)

    def settings(self, args):
        self.light = args.light
        self.fill = args.fill
        self.line = args.line
        self.transparent = args.transparent
        self.parallel = args.parallel
        self.animate = False
        self.speed = args.speed

    def toggle_light(self):
        self.light = not self.light
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
    def animation_speed(self, factor ):   
        self.speed *= factor


    def tick(self, dt):
        """
        invoked from Interactivity handlers on_idle as this is not getting those notifications

        hmm better way to prevent this being called too often ?
        """
        if not self.animate:return
        global count
        count += 1  
        self.scene.view(count, self.speed)
        self.frame.redraw() 

    def on_init(self):
        """
        glumpy.Figure.on_init sets up the RGB lights before dispatch to all figures
        """
        if self.light:
            self.lights.setup()
            log.info("on_init lights\n%s" % str(self.lights))
        pass

    def push(self):
        """
        Need to read the sequence of transformations backwards
        """
        trackball = self.scene.trackball
        camera = self.scene.camera
        view = self.scene.view

        scale = self.scale
        distance = view.distance


        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glPushMatrix()
        gl.glLoadIdentity ()


        lrbtnf = camera.lrbtnf 
        if self.parallel:
            gl.glOrtho ( *lrbtnf )
        else:
            gl.glFrustum ( *lrbtnf )
        pass
        
        # scaling here can see inner volumes, but not outer ones
        #gl.glScalef(1./scale, 1./scale, 1./scale)   # does nothing for scaled mode


        gl.glMatrixMode (gl.GL_MODELVIEW)
        gl.glPushMatrix()
        gl.glLoadIdentity ()

        # scaling in MODELVIEW transform (first in code) 
        # rather than first thing in PROJECTION transform (last in code)
        # succeeds to get lights under control because light positions
        # are stored in eye space, after the MODELVIEW transform is applied

        gl.glTranslate ( *trackball.xyz )

        gl.glScalef(1./scale, 1./scale, 1./scale)   

        gl.glMultMatrixf (trackball._matrix )   # rotation only 
        
        #gl.glTranslate ( 0, 0, -distance )
        #glut.glutWireSphere( scale*trackball._TRACKBALLSIZE,10,10)
        #gl.glTranslate ( 0, 0, +distance )


        if self.lookat:
            glu.gluLookAt( *view.eye_look_up )   
            # eye/look/up in world frame define the camera transform, 
            # translating eye to origin and rotating look to be along -z
            # NB no scaling, still world distances 

        if self.light:
            self.lights.position()   # reset positions following changes to MODELVIEW matrix ?


        

        gl.glPushMatrix()
        gl.glTranslate ( *view.look[:3] )
        glut.glutWireSphere( scale,10,10)
        gl.glPopMatrix()


        gl.glPushMatrix()
        gl.glTranslate ( *view.eye[:3] )
        glut.glutWireSphere( distance,10,10)
        gl.glPopMatrix()


    def unproject(self, x, y):
        """
        :param x: screen coordinates 
        :param y:

        #. gluUnProject needs window_z value,  z=0(near), z=1(far), z=depth(inbetween)
        #. read z from depth buffer for the xy 

        """
        self.push()

        pixels = gl.glReadPixelsf(x, y, 1, 1, gl.GL_DEPTH_COMPONENT ) # width,height 1,1  

        z = pixels[0][0]
        window_xyz = (x,y,z)
        if not (0 <= z <= 1): log.warn("unexpectd z buffer read %s " %  str(window_xyz))

        click = glu.gluUnProject( *window_xyz ) # click point in world frame       

        #log.debug("unproject %s => %s " % (str(window_xyz),str(click))) 

        geometry = self.scene.geometry
        f = geometry.find_bbox_solid( click )
        #log.info("find_bbox_solid %s yields %s solids %s " % (str(click), len(f), str(f)))
 
        view = self.scene.view
        eye,look,up = np.split(view.eye_look_up, 3)  # all world frame

        solids = sorted([geometry.solids[_] for _ in f],key=lambda _:_.extent, reverse=False) # presented from bottom so dont reverse

        self.annotate = [] 
        for solid in solids:
            log.info(solid)
            w2m = solid.world2model
            #log.info("click %s eye %s look %s " % (w2m(click),w2m(eye),w2m(look)) )
            self.annotate.append(repr(solid))
        pass
        self.pop()
        self.frame.redraw() 


    def pop(self):
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glPopMatrix()
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glPopMatrix()


    def on_draw(self):
        self.frame.lock()
        self.frame.draw()

        self.push() # matrices

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

        if len(self.annotate) > 0:
            self.text(self.annotate)

        self.frame.unlock()


    



if __name__ == '__main__':
    pass



