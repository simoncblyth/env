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

from env.graphics.pipeline.unit_transform import UnitTransform, KeyView
from env.graphics.pipeline.view_transform import ViewTransform
from env.geant4.geometry.collada.daegeometry import DAEGeometry
from env.graphics.transformations.transformations import quaternion_from_matrix

class FrameHandler(object):
    def __init__(self, frame, mesh, trackball, fill=True, line=True, transparent=True ):
        self.frame = frame
        self.mesh = mesh
        self.trackball = trackball
        self.fill = fill
        self.line = line
        self.transparent = transparent
        pass
        frame.push(self)     # event notification hookup

    def toggle_fill(self):
        self.fill = not self.fill
    def toggle_line(self):
        self.line = not self.line
    def toggle_transparent(self):
        self.transparent = not self.transparent
    def toggle_parallel(self):
        self.trackball.parallel = not self.trackball.parallel
 
    def on_draw(self):
        self.frame.lock()
        self.frame.draw()

        self.trackball.push()  # sets up the matrices

        if self.fill:
            if self.transparent:
                gl.glEnable (gl.GL_BLEND)
                gl.glBlendFunc ( gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
            pass

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

        self.trackball.pop()
        self.frame.unlock()





class FigHandler(object):
    def __init__(self, fig, frame_handler, trackball, light=True, fullscreen=False ):
        self.fig = fig
        self.frame_handler = frame_handler
        self.trackball = trackball
        self.light = light
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
        """
        Enabling lights 0,1,2 gives acceptable lighting 
        with a white background. Enabling them singly 
        gives nasty red/green/blue lighting and background.

        glumpy.Figure.on_init sets up the RGB lights before dispatch
        to all figures
        """
        if self.light:
            pass
            gl.glEnable (gl.GL_LIGHTING) # with just this line tis very dark, but no nasty red
            gl.glEnable (gl.GL_LIGHT0)  
            gl.glEnable (gl.GL_LIGHT1)  
            gl.glEnable (gl.GL_LIGHT2)   
        pass
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
        #print "on_mouse_drag button %s " % button  # perhaps avoid modal interface 
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



class MyTrackball(gp.Trackball):
    def __init__(self, thetaphi=(0,0), xyz=(0,0,3), extent=1., yfov=50, near=0.01, far=10. , parallel=False, matrix=None, scale=False, lookat=False, eye=None, look=None, up=None):
        ''' Build a new trackball with specified view '''

        self.lookat = lookat 
        self.eye = eye
        self.look = look
        self.up = up
        self.extent_factor = 1./extent

        self._count = 0 
        self._matrix= matrix

        if matrix is None: 
            self._rotation = [0,0,0,1]
            theta, phi = thetaphi
            self._set_orientation(theta,phi)
            self._x =  xyz[0]
            self._y =  xyz[1]
            self._z =  xyz[2]
        else:
            self.matrix = matrix # setter populates the rest  


        self._RENORMCOUNT = 97
        self._TRACKBALLSIZE = 0.8 
        self._yfov = yfov
        self._near = near
        self._far = far

        self.nearclip = 1e-4,1e6
        self.farclip = 1e-4,1e6

        self.parallel = parallel
        self.scale = scale

        # vestigial
        self.zoom = 0    
        self.distance = 0 


    def __repr__(self):
        return "yfov %3.1f near %4.1f far %4.1f x %4.1f y %4.1f z%4.1f theta %3.1f phi %3.1f" % \
            (self._yfov, self._near, self._far, self._x, self._y, self._z, self.theta, self.phi )


    def _get_height(self):
        viewport = gl.glGetIntegerv(gl.GL_VIEWPORT)
        return float(viewport[3])
    height = property(_get_height)

    def _get_width(self):
        viewport = gl.glGetIntegerv(gl.GL_VIEWPORT)
        return float(viewport[2])
    width = property(_get_width)

    def _get_aspect(self):
        viewport = gl.glGetIntegerv(gl.GL_VIEWPORT)
        return float(viewport[2])/float(viewport[3])
    aspect = property(_get_aspect)


    def zoom_to (self, x, y, dx, dy):
        """
        Zoom trackball by a factor dy 
        Changed from glumpy original _zoom to _distance
        so this is now a translation in z direction
        """
        self._z += -5*dy/self.height

    def pan_to (self, x, y, dx, dy):
        ''' Pan trackball by a factor dx,dy '''
        self._x += 3*dx/self.width
        self._y += 3*dy/self.height



    def near_to (self, x, y, dx, dy):
        ''' Change near clipping '''
        self.near += 5*dy/self.height

    def far_to (self, x, y, dx, dy):
        ''' Change far clipping '''
        self.far += 5*dy/self.height

    def yfov_to (self, x, y, dx, dy):
        ''' Change yfov '''
        self.yfov += 50*dy/self.height


    def _get_near(self):
        return self._near
    def _set_near(self, near):
        self._near = np.clip(near, self.nearclip[0], self.nearclip[1])
    near = property(_get_near, _set_near)

    def _get_far(self):
        return self._far
    def _set_far(self, far):
        self._far = np.clip(far, self.farclip[0],self.farclip[1])
    far = property(_get_far, _set_far)

    def _get_yfov(self):
        return self._yfov
    def _set_yfov(self, yfov):
        self._yfov = np.clip(yfov,5.,175.)
    yfov = property(_get_yfov, _set_yfov)


    def _get_matrix(self):
        return self._matrix
    def _set_matrix(self, matrix):
        self._matrix = matrix
        xyz = matrix[:3,3]
        self._x = xyz[0]
        self._y = xyz[1]
        self._z = -xyz[2]  # maybe need to negate
        q = quaternion_from_matrix(matrix)   
        self._rotation = [q[3],q[0],q[1],q[2]]  # different quaternion rep

    matrix = property(_get_matrix, _set_matrix)

    def _get_lrbtnf(self):
        """
        ::
                   . | 
                .    | top 
              +------- 
                near |
                     |
                   
        """
        aspect = self.aspect
        near = self._near
        far = self._far

        top = near * math.tan(self._yfov*0.5*math.pi/180.0)  
        bottom = -top
        left = aspect * bottom
        right = aspect * top 

        return left,right,bottom,top,near,far 

    lrbtnf = property(_get_lrbtnf)

    def gluLookAt(self):
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
        eye, look, up = self.eye, self.look, self.up
        #print "gluLookAt eye %s look %s up %s " % (str(eye),str(look),str(up))
        glu.gluLookAt(eye[0],eye[1],eye[2],look[0],look[1],look[2],up[0],up[1],up[2])


    def push(self):
        gl.glMatrixMode (gl.GL_MODELVIEW)
        gl.glPushMatrix()
        gl.glLoadIdentity ()

        if self.lookat:
            extent_factor = self.extent_factor
            gl.glScalef(extent_factor, extent_factor, extent_factor)  
            self.gluLookAt()
        else:
            gl.glTranslate (self._x, self._y, -self._z )
            gl.glMultMatrixf (self._matrix)


        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glPushMatrix()
        gl.glLoadIdentity ()

        lrbtnf = self.lrbtnf 
        #print "lrbtnf %s " % str(lrbtnf) 
        if self.parallel:
            gl.glOrtho ( *lrbtnf )
        else:
            gl.glFrustum ( *lrbtnf )
        pass

    def pop(void):
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glPopMatrix()
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glPopMatrix()







def compose_arg( arg, default, f_ ):
    """  
    Allows partial arg specification for comma delimited string arguments, 
    the rest coming from the defaults.
    """
    arg = f_(arg)
    default = f_(default)
    if len(arg) == len(default):
        ret = arg
    else:
        ret = arg + default[len(arg):]
    assert len(ret) == len(default), (ret, default)
    return ret


def parse_args(doc):
    import argparse
    defaults = {}
    #defaults['nodes']="3153:12230"
    defaults['nodes']="5000:5100"   # some PMTs for quick testing

    #defaults['size']="1440,852"
    defaults['size']="640,480"

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
    parser.add_argument(     "--near",  default=0.1, help="Initial near. %(default)s", type=float)
    parser.add_argument(     "--far",  default=5., help="Initial far. %(default)s", type=float)
    parser.add_argument(     "--thetaphi",  default="0,0", help="Initial theta,phi. %(default)s", type=str)
    parser.add_argument(     "--xyz",  default="0,0,3", help="Initial viewpoint in canonical -1:1 cube coordinates %(default)s", type=str)
    parser.add_argument(     "--parallel", action="store_true", help="Parallel projection, aka orthographic." )
    parser.add_argument(     "--fullscreen", action="store_true", help="Start in fullscreen mode." )

    # target based positioning mode switched on by presence of target 
    parser.add_argument("-t","--target", default=None,     help="Node index of solid on which to focus",type=str)
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

    return args




def backburner(args, geometry):
    target = None
    if args.target:
        target = geometry.find_solid(args.target) 
        if target is None:
           log.warn("failed to find target %s fallback to scale mode " % args.target )
        else:
           log.info("target : %s " % (target) )

    if target is None:
        scale, matrix = True, None
    else:
        scale = False
        unit = UnitTransform(geometry.get_bounds(target.index))
        print unit
        key  = KeyView( args.eye, args.look, args.up, unit )
        view = ViewTransform( *key._eye_look_up )
        print view
        matrix = view.matrix


def create_trackball( args, geometry ):
    """

    #. scaling the VBO coordinates seems wrong, that is too early to scale as 
       want to use world coordinates to locate viewpoints

    #. presenting geometry in a coordinate frame with 0,0,0 at the center
       of the presented vertices and extents from -1 to 1 is useful, as x,y,z
       has some meaning as you move around 
      
    """
    kwa = {}
    kwa['thetaphi'] = args.thetaphi
    kwa['xyz'] = args.xyz
    kwa['yfov'] = args.yfov
    kwa['near'] = args.near
    kwa['far'] = args.far
    kwa['parallel'] = args.parallel

    if args.target is None:
        kwa['matrix'] = None
        kwa['scale'] = True
        kwa['lookat'] = False
    else:

        lower, upper, extent = geometry.mesh.bounds_extent

        kwa['scale'] = False
        kwa['lookat'] = True
        kwa['extent'] = extent

        unit = UnitTransform([lower,upper])
        key  = KeyView( args.eye, args.look, args.up, unit )
        eye, look, up = key._eye_look_up

        with printoptions(precision=3, suppress=True, strip_zeros=False):
             print geometry.mesh.smry()
             print key 

        kwa['eye'] = eye
        kwa['look'] = look
        kwa['up'] = up

    trackball = MyTrackball(**kwa)
    return trackball


def main():
    args = parse_args(__doc__)
    geometry = DAEGeometry(args.nodes, path=args.path)
    geometry.flatten()

    trackball = create_trackball( args, geometry )

    fig = gp.Figure(size=args.size)
    frame = fig.add_frame(size=args.frame)
    vbo = geometry.make_vbo(scale=trackball.scale, rgba=args.rgba )
    mesh = gp.graphics.VertexBuffer( vbo.data, vbo.faces )

    frame_handler = FrameHandler( frame, mesh, trackball, fill=args.fill, line=args.line, transparent=args.transparent )
    fig_handler = FigHandler(fig, frame_handler, trackball, light=args.light, fullscreen=args.fullscreen )

    gp.show()


if __name__ == '__main__':
    main()

