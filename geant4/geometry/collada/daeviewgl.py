#!/usr/bin/env python
"""

"""
import os, logging
log = logging.getLogger(__name__)

import numpy as np
import glumpy as gp  
import OpenGL.GL as gl

from glumpy.figure import Frame

from env.geant4.geometry.collada.daegeometry import DAEGeometry 
from npcommon import printoptions, load_obj


class VBO(object):
    @classmethod
    def from_obj(cls, filename="/usr/local/env/graphics/glumpy/glumpy/demos/triceratops.obj"): 
        vertices, normals, faces = load_obj(filename)
        return cls(vertices, normals, faces)

    @classmethod
    def from_dae(cls, arg, scale=False):
        dg = DAEGeometry(arg)
        dg.flatten()

        if scale:
            vertices = (dg.mesh.vertices - dg.mesh.center)/dg.mesh.extent
        else:
            vertices = dg.mesh.vertices

        normals = dg.mesh.normals
        faces = dg.mesh.triangles
        return cls(vertices, normals, faces)

    def __init__(self, vertices, normals, faces):

        V = np.zeros(len(vertices), [('position', np.float32, 3), 
                                     ('color', np.float32, 3), 
                                     ('normal',   np.float32, 3)])
        V['position'] = vertices
        V['color'] = (vertices+1)/2.0
        V['normal'] = normals

        self.V = V
        self.faces = faces

    def vbo(self):
        return gp.graphics.VertexBuffer( self.V, self.faces )

    def __repr__(self):
        with printoptions(precision=3, suppress=True, strip_zeros=False):
            return "\n".join([
                   "position",str(self.V['position']),
                   "color",str(self.V['color']),
                   "normal",str(self.V['normal']),
                   "faces",str(self.faces),
                   ])





class DAEFrame(Frame):
    def __init__(self, *args, **kwa):
        log.info("DAEFrame __init__")
        Frame.__init__(self, *args, **kwa)

    def on_draw(self):
        log.info("DAEFrame on_draw")
        self.lock()
        self.draw()
        self.trackball.push()

        gl.glEnable( gl.GL_POLYGON_OFFSET_FILL )
        gl.glPolygonOffset (1, 1)
        gl.glPolygonMode( gl.GL_FRONT_AND_BACK, gl.GL_FILL )

        self.mesh.draw( gl.GL_TRIANGLES, "pnc" )

        gl.glDisable( gl.GL_POLYGON_OFFSET_FILL )
        gl.glPolygonMode( gl.GL_FRONT_AND_BACK, gl.GL_LINE )
        gl.glEnable( gl.GL_BLEND )
        gl.glEnable( gl.GL_LINE_SMOOTH )
        gl.glColor( 0.0, 0.0, 0.0, 0.5 )

        self.mesh.draw( gl.GL_TRIANGLES, "p" )

        gl.glDisable( gl.GL_BLEND )
        gl.glDisable( gl.GL_LINE_SMOOTH )

        self.trackball.pop()
        self.unlock()


#DAEFrame.register_event_type('on_draw')


class DAEFigure(gp.Figure):
    def __init__(self, *args, **kwa):
        log.info("DAEFigure __init__")
        gp.Figure.__init__(self, *args, **kwa)

    def on_init(self):
        log.info("DAEFigure on_init")
        gl.glLightfv (gl.GL_LIGHT0, gl.GL_DIFFUSE, (1.0, 1.0, 1.0, 1.0))
        gl.glLightfv (gl.GL_LIGHT0, gl.GL_AMBIENT, (0.3, 0.3, 0.3, 1.0))
        gl.glLightfv (gl.GL_LIGHT0, gl.GL_SPECULAR,(0.0, 0.0, 0.0, 0.0))
        gl.glLightfv (gl.GL_LIGHT0, gl.GL_POSITION,(2.0, 2.0, 2.0, 0.0))
        gl.glEnable (gl.GL_LIGHTING)
        gl.glEnable (gl.GL_LIGHT0)

    def on_mouse_drag(self, x,y,dx,dy,button):
        log.info("DAEFigure on_mouse_drag")
        self.trackball.drag_to(x,y,dx,dy)
        self.redraw()

    def on_draw(self):
        log.info("DAEFigure on_draw")
        self.clear(0.85,0.85,0.85,1)

    def add_frame(self, size = (0.9,0.9), spacing = 0.025, aspect=None):
        log.info("DAEFigure add_frame")
        return DAEFrame(self, size=size, spacing=spacing, aspect=aspect)


def main():
    import sys
    logging.basicConfig(level=logging.INFO)
    
    vbo = VBO.from_dae("4998:6500", scale=True)

    fig = DAEFigure((1024,768))
    trackball = gp.Trackball( 65, 135, 1.0, 2.5 )
    fig.trackball = trackball

    frame = fig.add_frame(size=(1,1))
    frame.mesh = vbo.vbo()
    frame.trackball = trackball

    frame.on_draw()

    gp.show()


if __name__ == '__main__':
    main()

