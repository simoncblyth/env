#!/usr/bin/env python
"""
DAEVIEWGL
==========

::

    daeviewgl.py -n 4998:6000

      # default includes lights, fill with transparency 

    daeviewgl.py -n 4998:6000 --line

      # adding wireframe lines slows rendering significantly

    daeviewgl.py -n 4998 --nofill

       # without polygon fill the lighting/transparency has no effect

    daeviewgl.py -n 4998 --nofill 

       # blank white 

    daeviewgl.py -n 4900:5000,4815 --notransparent

       # see the base of the PMTs poking out of the cylinder when transparency off

    daeviewgl.py -n 4900:5000,4815 --rgba .7,.7,.7,0.5

       # changing colors, especially alpha has a drastic effect on output

    daeviewgl.py -n 4900:5000,4815 --ball 90,0,2,3

       # primitive initial position control using trackball arguments, theta,phi,zoom,distance

    daeviewgl.py -n 3153:6000

       # inside the pool, 2 ADs : navigation is a challenge, its dark inside

    daeviewgl.py -n 6070:6450
  
       # AD structure, shows partial radial shield

    daeviewgl.py -n 6480:12230 

       # pool PMTs, AD support, scaffold?    when including lots of volumes switching off lines is a speedup

    daeviewgl.py -n 12221:12230 
 
       # rad slabs

    daeviewgl.py -n 2:12230 

       # full geometry, excluding only boring (and large) universe and rock 

    daeviewgl.py -n 3153:12230

       # skipping universe, rock and RPC makes for easier inspection inside the pool

    daeviewgl.py  -n 3153:12230 -t 5000 --eye="-2,-2,-2"

       # target mode, presenting many volumes but targeting one and orienting viewpoint with 
       # respect to the target using units based on the extent of the target and axis directions
       # from the world frame
       #
       # long form --eye="..." is needed as the value starts with "-"


Parallel/Orthographic projection
----------------------------------

In parallel projection, there is effectively no z direction (its
as if viewing from infinity) so varying z has no effect.  Instead
to control view change near and yfov.  It can be difficult 
to "enter" geometry while in parallel, to do so try very small yfov (5 degrees) 
and vary near.


Controls
---------

* SPACE and mouse/trackpad up/down to translate z (in-out of direction of view), 
  also a non-modal alternative is click+two-finger drag 
* TAB and mouse/trackpad to translate x,y (left-right,up-down)
* **S** toggle fullscreen 
* **F** toggle fill
* **L** toggle line
* **T** toggle transparent
* **P** toggle parallel projection (aka orthographic)
* **N** toggle near mode, where mouse/trackpad changes near clipping plane 
* **A** toggle far mode, where mouse/trackpad changes far clipping plane 
* **Y** toggle yfov mode, where mouse/trackpad changes field of view 
  also **UP** and **DOWN** arrow yets changes yfov in 5 degrees increments within
  range of 5 to 175 degrees. Extreme wideangle is useful when using parallel projection. 


Observations
--------------

#. toggling between parallel/perspective can be dis-orienting, change near/yfov to get the desired view  


Issues
--------

#. somehow near can end up getting very small and causing problems that look like far clipping

TODO
----

#. clipping plane controls, that are fixed in world coordinates

#. external positioning control, lookat/eye/up   
#. placemarks
#. coloring by material


"""
import os, sys, math, logging
log = logging.getLogger(__name__)

import numpy as np
from npcommon import printoptions

import glumpy as gp  
from glumpy.window import key
import OpenGL.GL as gl
import OpenGL.GLU as glu

from daetrackball import DAETrackball


from env.graphics.pipeline.unit_transform import UnitTransform, KeyView
from env.graphics.pipeline.view_transform import ViewTransform
from env.geant4.geometry.collada.daegeometry import DAEGeometry
from env.graphics.transformations.transformations import quaternion_from_matrix



class FrameHandler(object):
    def __init__(self, frame, mesh, trackball, fill=True, line=True, transparent=True, light=True):
        self.frame = frame
        self.mesh = mesh
        self.trackball = trackball
        self.fill = fill
        self.line = line
        self.transparent = transparent
        self.light = light
        pass
        frame.push(self)     # event notification hookup


    # toggles invoked by FigHandler
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

        if trackball.lookat:
            refextent = trackball.refextent
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





class FigHandler(object):
    """
    Keep this for handling interactivity, **NOT** graphics    
    """
    def __init__(self, fig, frame_handler, trackball, fullscreen=False ):
        self.fig = fig
        self.frame_handler = frame_handler
        self.trackball = trackball
        # 
        self.zoom_mode = False
        self.pan_mode = False
        self.near_mode = False
        self.far_mode = False
        self.yfov_mode = False

        if fullscreen:
            self.toggle_fullscreen()
        pass
        fig.push(self)   # event notification

    def _get_title(self):
        return "%s" % repr(self.trackball)
    title = property(_get_title)     

    def redraw(self):
        self.fig.window.set_title(self.title)
        self.fig.redraw()

    def toggle_fullscreen(self):
        if self.fig.window.get_fullscreen():
            self.fig.window.set_fullscreen(0)
        else:
            self.fig.window.set_fullscreen(1)
 
    def on_init(self):
        self.fig.window.set_title(self.title)

    def on_draw(self):
        self.fig.clear(0.85,0.85,0.85,1)  # seems to have no effect even when lighting disabled

    def on_key_press(self, symbol, modifiers):
        if   symbol == key.ESCAPE: sys.exit();
        elif symbol == key.SPACE: self.zoom_mode = True
        elif symbol == key.TAB: self.pan_mode = True
        elif symbol == key.N: self.near_mode = True
        elif symbol == key.A: self.far_mode = True
        elif symbol == key.Y: self.yfov_mode = True
        elif symbol == key.UP: self.trackball.yfov += 5.
        elif symbol == key.DOWN: self.trackball.yfov += -5.
        elif symbol == key.S: self.toggle_fullscreen()
        elif symbol == key.L: self.frame_handler.toggle_line()
        elif symbol == key.F: self.frame_handler.toggle_fill()
        elif symbol == key.T: self.frame_handler.toggle_transparent()
        elif symbol == key.P: self.frame_handler.toggle_parallel()
        else:
            print "no action for on_key_press with symbol 0x%x " % symbol
        pass 
        self.redraw()

    def on_key_release(self,symbol, modifiers):
        if   symbol == key.SPACE: self.zoom_mode = False
        elif symbol == key.TAB: self.pan_mode = False
        elif symbol == key.N: self.near_mode = False
        elif symbol == key.A: self.far_mode = False
        elif symbol == key.Y: self.yfov_mode = False
        else:
            print "no action for on_key_release with symbol 0x%x " % symbol
        pass
        self.redraw()

    def on_mouse_drag(self,x,y,dx,dy,button):
        two_finger_zoom = button == 8  # NB zoom is a misnomer, this is translating eye coordinate z
        if   self.zoom_mode or two_finger_zoom: self.trackball.zoom_to(x,y,dx,dy)
        elif self.pan_mode: self.trackball.pan_to(x,y,dx,dy)
        elif self.near_mode: self.trackball.near_to(x,y,dx,dy)
        elif self.far_mode: self.trackball.far_to(x,y,dx,dy)
        elif self.yfov_mode: self.trackball.yfov_to(x,y,dx,dy)
        else: 
            self.trackball.drag_to(x,y,dx,dy)
        pass
        self.redraw()

    def on_mouse_press(self, x, y, button):
        print 'Mouse button pressed (x=%.1f, y=%.1f, button=%d)' % (x,y,button)

    def on_mouse_release(self, x, y, button):
        print 'Mouse button released (x=%.1f, y=%.1f, button=%d)' % (x,y,button)

    def on_mouse_motion(self, x, y, dx, dy):
        pass
        #print 'Mouse motion (x=%.1f, y=%.1f, dx=%.1f, dy=%.1f)' % (x,y,dx,dy)  # get lots of these

    def on_mouse_scroll(self, x, y, dx, dy):
        print 'Mouse scroll (x=%.1f, y=%.1f, dx=%.1f, dy=%.1f)' % (x,y,dx,dy)   # none of these

    #def on_idle(self, dt):
    #    print 'Idle event ', dt








def parse_args(doc):
    import argparse
    defaults = {}
    defaults['nodes']="3153:12230"
    #defaults['nodes']="5000:5100"   # some PMTs for quick testing

    defaults['size']="1440,852"
    #defaults['size']="640,480"

    parser = argparse.ArgumentParser(doc)
    parser.add_argument("-n","--nodes", default=defaults['nodes'],   help="DAENode.getall node(s) specifier %(default)s",type=str)
    parser.add_argument(     "--size", default=defaults['size'], help="Pixel size  %(default)s", type=str)
    parser.add_argument(     "--path", default=os.environ['DAE_NAME'], help="Path of geometry file  %(default)s",type=str)

    parser.add_argument(  "--line", dest="line", action="store_true", help="Switch on line mode polygons  %(default)s" )

    parser.add_argument("--nolight",dest="light", action="store_false", help="Inhibit light setup  %(default)s" )
    parser.add_argument("--nofill", dest="fill", action="store_false", help="Inhibit fill mode polygons  %(default)s" )
    parser.add_argument("--notransparent",dest="transparent", action="store_false", help="Inhibit transparent fill  %(default)s" )

    parser.add_argument("--rgba",  default=".7,.7,.7,.5", help="RGBA color of geometry, the alpha has a dramatic effect  %(default)s",type=str)
    parser.add_argument("--frame",  default="1,1", help="Viewport framing  %(default)s",type=str)

    parser.add_argument("-l","--loglevel", default="INFO", help="INFO/DEBUG/WARN/..   %(default)s")  
    
    parser.add_argument(     "--yfov",  default=50., help="Initial vertical field of view in degrees. %(default)s", type=float)
    parser.add_argument(     "--near",  default=0.001, help="Initial near. %(default)s", type=float)
    parser.add_argument(     "--far",  default=100., help="Initial far. %(default)s", type=float)
    parser.add_argument(     "--thetaphi",  default="0,0", help="Initial theta,phi. %(default)s", type=str)
    parser.add_argument(     "--xyz",  default="0,0,3", help="Initial viewpoint in canonical -1:1 cube coordinates %(default)s", type=str)
    parser.add_argument(     "--parallel", action="store_true", help="Parallel projection, aka orthographic." )
    parser.add_argument(     "--fullscreen", action="store_true", help="Start in fullscreen mode." )

    # target based positioning mode switched on by presence of target 
    parser.add_argument("-t","--target", default=None,     help="Node specification of solid on which to focus or empty string for all",type=str)
    parser.add_argument("-e","--eye",   default="-2,0,0", help="Eye position",type=str)
    parser.add_argument("-a","--look",  default="0,0,0",   help="Lookat position",type=str)
    parser.add_argument("-u","--up",   default="0,0,1", help="Eye position",type=str)


    # not yet implemented
    parser.add_argument("-F","--noflip",  dest="flip", action="store_false", default=True, help="Pixel y flip.")
    parser.add_argument("-s","--pscale", default=1., help="Parallel projection, scale.", type=float  )
    parser.add_argument("-i","--interactive", action="store_true", help="Interative Mode")
    parser.add_argument("-j","--jump", default=None, help="Animated transition to another node.")  

    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.loglevel))
    
    fvec_ = lambda _:map(float, _.split(","))
    ivec_ = lambda _:map(int, _.split(","))

    args.thetaphi = fvec_(args.thetaphi) 
    args.xyz = fvec_(args.xyz) 

    args.frame = fvec_(args.frame) 
    args.rgba = fvec_(args.rgba) 
    args.eye = fvec_(args.eye) 
    args.look = fvec_(args.look) 
    args.up = fvec_(args.up) 
    args.size = ivec_(args.size) 


    if args.target is None:
       if args.near < 1.:
           args.near = 1. 
           log.warn("amending near to %s for non-target mode" % args.near)

    return args



class Scene(object):
    """
    #. scaling the VBO coordinates seems wrong, that is too early to scale as 
       want to use world coordinates to locate viewpoints

    #. presenting geometry in a coordinate frame with 0,0,0 at the center
       of the presented vertices and extents from -1 to 1 is useful, as x,y,z
       has some meaning as you move around 
    """  
    def __init__(self, args, geometry ):

        self.args = args
        self.geometry = geometry  
        self.mesh = geometry.mesh
        self.view = None

        self.kwa = {}
        self.configure_target()
        self.configure_base()
        if not self.target is None:
            self.configure_lookat() 

        self.trackball = DAETrackball(**self.kwa)

    def configure_target(self):
        if self.args.target is None:
            target = None
        else:
            target = self.geometry.find_solid(self.args.target) 
            assert target, "failed to find target for argument %s " % self.args.target
        self.target = target 

    def configure_base(self):
        args = self.args
        self.kwa['thetaphi'] = args.thetaphi
        self.kwa['xyz'] = args.xyz
        self.kwa['yfov'] = args.yfov
        self.kwa['near'] = args.near
        self.kwa['far'] = args.far
        self.kwa['parallel'] = args.parallel

    def configure_lookat(self):
        """
        Convert eye/look/up input parameters into world coordinates
        """
        lower, upper, extent = self.target.bounds_extent
        unit = UnitTransform([lower,upper])

        self.view  = KeyView( self.args.eye, self.args.look, self.args.up, unit )
        eye, look, up = self.view._eye_look_up

        self.kwa['lookat'] = True
        self.kwa['extent'] = extent
        self.kwa['eye'] = eye
        self.kwa['look'] = look
        self.kwa['up'] = up

    def dump(self):
        if self.view:
            print "view\n", self.view
        if self.mesh: 
            print "full mesh\n",self.mesh.smry()
        if self.target: 
            print "target mesh\n",self.target.smry()



def main():
    np.set_printoptions(precision=4, suppress=True)

    args = parse_args(__doc__)
    geometry = DAEGeometry(args.nodes, path=args.path)
    geometry.flatten()

    scene = Scene(args, geometry)
    scene.dump() 
    trackball = scene.trackball

    fig = gp.Figure(size=args.size)
    frame = fig.add_frame(size=args.frame)

    vbo = geometry.make_vbo(scale=not scene.target, rgba=args.rgba )
    print vbo
    mesh = gp.graphics.VertexBuffer( vbo.data, vbo.faces )

    frame_handler = FrameHandler( frame, mesh, trackball, fill=args.fill, line=args.line, transparent=args.transparent, light=args.light )
    fig_handler = FigHandler(fig, frame_handler, trackball, fullscreen=args.fullscreen )

    gp.show()


if __name__ == '__main__':
    main()

