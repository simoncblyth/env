#!/usr/bin/env python
"""
DAEVIEWGL
==========

::

    daeviewgl.py -n 4998:6000

      # default includes lights, fill with transparency and wireframe lines

    daeviewgl.py -n 4998:6000 --noline

      # without the wireframe lines, rendering is a bit faster 
      # (hmm surely there is a quicker way to wireframe)

    daeviewgl.py -n 4998 --nofill

       # wireframe only, lighting/transparency has no effect

    daeviewgl.py -n 4998 --nofill --noline

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

    daeviewgl.py -n 6480:12230  --noline

       # pool PMTs, AD support, scaffold?    when including lots of volumes switching off lines is a speedup

    daeviewgl.py -n 12221:12230 
 
       # rad slabs

    daeviewgl.py -n 2:12230  --noline

       # full geometry 




Controls
---------

* SPACE and mouse/trackpad up/down to translate z (in-out of direction of view)
* TAB and mouse/trackpad to translate x,y (left-right,up-down)
* **S** toggle fullscreen 
* **F** toggle fill
* **L** toggle line
* **T** toggle transparent

Issues
--------

#. panning is too fast, needs to adapt to the scale of whats displayed

TODO
----

#. external positioning control, lookat/eye/up   
#. placemarks
#. coloring by material


"""
import os, sys, logging
log = logging.getLogger(__name__)

import numpy as np
import glumpy as gp  
import OpenGL.GL as gl

from glumpy.figure import Frame

from env.geant4.geometry.collada.daegeometry import DAEGeometry 
from npcommon import printoptions


class VBO(object):
    @classmethod
    def from_dae(cls, arg, path=None, scale=False, rgba=(0.7,0.7,0.7,0.5)):
        dg = DAEGeometry(arg, path=path)
        dg.flatten()

        if scale:
            vertices = (dg.mesh.vertices - dg.mesh.center)/dg.mesh.extent
        else:
            vertices = dg.mesh.vertices

        normals = dg.mesh.normals
        faces = dg.mesh.triangles
        return cls(vertices, normals, faces, rgba )

    def __init__(self, vertices, normals, faces, rgba ):

        nvert = len(vertices)
        data = np.zeros(nvert, [('position', np.float32, 3), 
                                ('color',    np.float32, 4), 
                                ('normal',   np.float32, 3)])
        data['position'] = vertices
        data['color'] = np.tile( rgba, (nvert, 1))
        data['normal'] = normals

        self.data = data
        self.faces = faces

    def __repr__(self):
        with printoptions(precision=3, suppress=True, strip_zeros=False):
            return "\n".join([
                   "position",str(self.data['position']),
                   "color",str(self.data['color']),
                   "normal",str(self.data['normal']),
                   "faces",str(self.faces),
                   ])





class FrameHandler(object):
    def __init__(self, frame, mesh, trackball, fill=True, line=True, transparent=True ):
        self.frame = frame
        self.mesh = mesh
        self.trackball = trackball
        self.fill = fill
        self.line = line
        self.transparent = transparent
        frame.push(self)

    def toggle_fill(self):
        self.fill = not self.fill
    def toggle_line(self):
        self.line = not self.line
    def toggle_transparent(self):
        self.transparent = not self.transparent

         

    def on_draw(self):
        """
        Transparency added according to 

        * http://www.opengl.org/archives/resources/faq/technical/transparency.htm

        """
        log.debug("DAEFrame on_draw")
        self.frame.lock()
        self.frame.draw()

        self.trackball.push()

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
            gl.glColor( 0., 0., 0., 0.5 )      # difficult to see lines from some directions, unless black

            self.mesh.draw( gl.GL_TRIANGLES, "p" )  # position

            gl.glDisable( gl.GL_BLEND )
            gl.glDisable( gl.GL_LINE_SMOOTH )

        self.trackball.pop()
        self.frame.unlock()





class FigHandler(object):
    def __init__(self, fig, frame_handler, trackball, light=True):
        self.fig = fig
        self.frame_handler = frame_handler
        self.trackball = trackball
        self.light = light
        self.zoom_mode = False
        self.pan_mode = False
        fig.push(self)

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
        """
        log.debug("DAEFigure on_init")
        if self.light:
            pass
            #gl.glLightfv (gl.GL_LIGHT0, gl.GL_DIFFUSE, (1.0, 1.0, 1.0, 1.0))
            #gl.glLightfv (gl.GL_LIGHT0, gl.GL_AMBIENT, (0.3, 0.3, 0.3, 1.0))
            #gl.glLightfv (gl.GL_LIGHT0, gl.GL_SPECULAR,(0.0, 0.0, 0.0, 0.0))
            #gl.glLightfv (gl.GL_LIGHT0, gl.GL_POSITION,(2.0, 2.0, 2.0, 0.0))
            gl.glEnable (gl.GL_LIGHTING) # with just this line tis very dark, but no nasty red
            gl.glEnable (gl.GL_LIGHT0)  
            gl.glEnable (gl.GL_LIGHT1)  
            gl.glEnable (gl.GL_LIGHT2)   

    def on_draw(self):
        log.debug("DAEFigure on_draw")
        self.fig.clear(0.85,0.85,0.85,1)  # seems to have no effect even when lighting disabled

    def on_key_press(self, symbol, modifiers):
        #print 'Key pressed (symbol=%s, modifiers=%s)'% (symbol,modifiers)
        if symbol == gp.window.key.ESCAPE:
            sys.exit();
        elif symbol == gp.window.key.SPACE:
            self.zoom_mode = True
        elif symbol == gp.window.key.TAB:
            self.pan_mode = True
        elif symbol == gp.window.key.S:
            self.toggle_fullscreen()
        elif symbol == gp.window.key.L:
            self.frame_handler.toggle_line()
        elif symbol == gp.window.key.F:
            self.frame_handler.toggle_fill()
        elif symbol == gp.window.key.T:
            self.frame_handler.toggle_transparent()
        else:
            print "no action for on_key_press with symbol %s " % symbol
        pass 
        self.fig.redraw()

    def on_key_release(self,symbol, modifiers):
        #print 'Key released (symbol=%s, modifiers=%s)'% (symbol,modifiers)
        if symbol == gp.window.key.SPACE:
            #print "released space"
            self.zoom_mode = False
        elif symbol == gp.window.key.TAB:
            #print "released tab"
            self.pan_mode = False

    def on_mouse_drag(self,x,y,dx,dy,button):
        #log.debug("DAEFigure on_mouse_drag")
        if self.zoom_mode:
            #log.info("on_mouse_drag trackball.zoom_to")
            self.trackball.zoom_to(x,y,dx,dy)
        elif self.pan_mode:
            #log.info("on_mouse_drag trackball.pan_to")
            self.trackball.pan_to(x,y,dx,dy)
        else: 
            #log.info("on_mouse_drag trackball.drag_to")
            self.trackball.drag_to(x,y,dx,dy)
        self.fig.redraw()

    def on_mouse_press(self, x, y, button):
        print 'Mouse button pressed (x=%.1f, y=%.1f, button=%d)' % (x,y,button)

    def on_mouse_release(self, x, y, button):
        print 'Mouse button released (x=%.1f, y=%.1f, button=%d)' % (x,y,button)

    def on_mouse_motion(self, x, y, dx, dy):
        pass
        #print 'Mouse motion (x=%.1f, y=%.1f, dx=%.1f, dy=%.1f)' % (x,y,dx,dy)

    def on_mouse_scroll(self, x, y, dx, dy):
        print 'Mouse scroll (x=%.1f, y=%.1f, dx=%.1f, dy=%.1f)' % (x,y,dx,dy)

    #def on_idle(self, dt):
    #    print 'Idle event ', dt





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
    defaults = dict(ball="65,135,1.,2.5",nodes="4900:4910")
    parser = argparse.ArgumentParser(doc)
    parser.add_argument("-n","--nodes", default=defaults['nodes'],   help="DAENode.getall node(s) specifier %(default)s",type=str)
    parser.add_argument("-z","--size", default="1024,768", help="Pixel size  %(default)s", type=str)
    parser.add_argument(     "--path", default=os.environ['DAE_NAME'], help="Path of geometry file  %(default)s",type=str)

    parser.add_argument("--nolight",dest="light", action="store_false", help="Inhibit light setup  %(default)s" )
    parser.add_argument("--nofill", dest="fill", action="store_false", help="Inhibit fill mode polygons  %(default)s" )
    parser.add_argument("--noline", dest="line", action="store_false", help="Inhibit line mode polygons  %(default)s" )
    parser.add_argument("--notransparent",dest="transparent", action="store_false", help="Inhibit transparent fill  %(default)s" )
    parser.add_argument("--rgba",  default=".7,.7,.7,.5", help="RGBA color of geometry, the alpha has a dramatic effect  %(default)s",type=str)
    parser.add_argument("--frame",  default="1,1", help="Viewport framing  %(default)s",type=str)
    parser.add_argument("-l","--loglevel", default="INFO", help="INFO/DEBUG/WARN/..   %(default)s")  
    
    parser.add_argument("--ball",  default=defaults['ball'], help="Trackball theta,phi,zoom,distance %(default)s",type=str)

    # not yet implemented
    parser.add_argument("-t","--target", default=None,     help="Node index of solid on which to focus",type=str)
    parser.add_argument("-a","--look",  default="0,0,0",   help="Lookat position",type=str)
    parser.add_argument("-e","--eye",   default="-2,0,0", help="Eye position",type=str)
    parser.add_argument("-u","--up",   default="0,0,1", help="Eye position",type=str)
    parser.add_argument("-f","--fov",  default=50., help="Vertical field of view in degrees.", type=float)
    parser.add_argument(     "--near",  default=1., help="Scale factor to apply to near distance, from eye to target node center.", type=float)
    parser.add_argument("-F","--noflip",  dest="flip", action="store_false", default=True, help="Pixel y flip.")
    parser.add_argument("-p","--parallel", action="store_true", help="Parallel projection, aka orthographic." )
    parser.add_argument("-s","--pscale", default=1., help="Parallel projection, scale.", type=float  )
    parser.add_argument("-i","--interactive", action="store_true", help="Interative Mode")
    parser.add_argument("-j","--jump", default=None, help="Animated transition to another node.")  

    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.loglevel))
    
    fvec_ = lambda _:map(float, _.split(","))
    ivec_ = lambda _:map(int, _.split(","))

    args.ball = compose_arg(args.ball, defaults['ball'], fvec_ ) 

    args.frame = fvec_(args.frame) 
    args.rgba = fvec_(args.rgba) 
    args.eye = fvec_(args.eye) 
    args.look = fvec_(args.look) 
    args.up = fvec_(args.up) 
    args.size = ivec_(args.size) 

    return args


class MyTrackball(gp.Trackball):
    def __init__(self, *args, **kwa):
        gp.Trackball.__init__(self, *args, **kwa)
    def _get_zoom(self):
        return self._zoom
    def _set_zoom(self, zoom):
        self._zoom = zoom
        #if self._zoom < .25: self._zoom = .25 
        #if self._zoom > 10: self._zoom = 10
    zoom = property(_get_zoom, _set_zoom,
                     doc='''Zoom factor''')


    def zoom_to (self, x, y, dx, dy):
        ''' Zoom trackball by a factor dy '''
        viewport = gl.glGetIntegerv(gl.GL_VIEWPORT)
        height = float(viewport[3])
        #self.zoom = self.zoom-5*dy/height
        self._distance = self._distance-5*dy/height

    def pan_to (self, x, y, dx, dy):
        ''' Pan trackball by a factor dx,dy '''
        self._x += dx*0.05
        self._y += dy*0.05





def main():
   
    args = parse_args(__doc__)
    vbo = VBO.from_dae(args.nodes, path=args.path, scale=True, rgba=args.rgba )


    
    fig = gp.Figure(size=args.size)
    frame = fig.add_frame(size=args.frame)

    mesh = gp.graphics.VertexBuffer( vbo.data, vbo.faces )
    trackball = MyTrackball( *args.ball )

    frame_handler = FrameHandler(frame, mesh, trackball, fill=args.fill, line=args.line, transparent=args.transparent )
    fig_handler = FigHandler(fig, frame_handler, trackball, light=args.light)


    gp.show()


if __name__ == '__main__':
    main()

