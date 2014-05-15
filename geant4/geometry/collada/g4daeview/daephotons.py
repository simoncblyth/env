#!/usr/bin/env python
"""
For checking using LXe example::

    g4daeview.sh -p lxe -g 1: -n1

    lxe-
    lxe-test   # sling some photons via RootZMQ

Interactive probing::

   (chroma_env)delta:g4daeview blyth$ ipython daechromaphotonlist.py -i

    In [12]: ddir = map(lambda _:np.linalg.norm(_), cpl.dir )


Hmm comingling gl code with pure numpy code 
is inconvenient as needs context to run. 

"""
import logging
import numpy as np
log = logging.getLogger(__name__)


import ctypes
import glumpy as gp
import OpenGL.GL as gl
import OpenGL.GLUT as glut

from env.graphics.color.wav2RGB import wav2RGB
from photons import Photons   # TODO: merge photons.Photons into my forked chroma.event.Photons
from daegeometry import DAEMesh 


class MyVertexBuffer(gp.graphics.VertexBuffer):
    """
    Inherit from glumpy VertexBuffer in order to experiment with the DrawElements 
    call : attempting for partial draws.
    """
    def __init__(self, *args ):
        gp.graphics.VertexBuffer.__init__(self, *args )

    def draw( self, mode=gl.GL_QUADS, what='pnctesf', offset=0, count=None ):
        """ 
        :param mode: primitive to draw
        :param what: attribute multiple choice by first letter
        :param offset: integer element array buffer offset, default 0
        :param count: number of elements, default None corresponds to all in self.indices

        Buffer offset default of 0 corresponds to glumpy original None, (ie (void*)0 )
        the integet value is converted with `ctypes.c_void_p(offset)`   
        allowing partial buffer drawing.

        * http://pyopengl.sourceforge.net/documentation/manual-3.0/glDrawElements.html
        * http://stackoverflow.com/questions/11132716/how-to-specify-buffer-offset-with-pyopengl
        * http://pyopengl.sourceforge.net/documentation/pydoc/OpenGL.arrays.vbo.html


        ====================  ==============
        type
        ====================  ==============
        GL_UNSIGNED_BYTE        0:255
        GL_UNSIGNED_SHORT,      0:65535
        GL_UNSIGNED_INT         0:4.295B
        ====================  ==============

        ===================   ====================================
           mode 
        ===================   ====================================
          GL_POINTS
          GL_LINE_STRIP
          GL_LINE_LOOP
          GL_LINES
          GL_TRIANGLE_STRIP
          GL_TRIANGLE_FAN
          GL_TRIANGLES
          GL_QUAD_STRIP
          GL_QUADS
          GL_POLYGON
        ===================   ====================================


        The what letters, 'pnctesf' define the meaning of the arrays via 
        enabling appropriate attributes.

        ==================  ==================   ================   =====
        gl***Pointer          GL_***_ARRAY          Att names         *
        ==================  ==================   ================   =====
         Color                COLOR                color              c
         EdgeFlag             EDGE_FLAG            edge_flag          e
         FogCoord             FOG_COORD            fog_coord          f
         Normal               NORMAL               normal             n
         SecondaryColor       SECONDARY_COLOR      secondary_color    s
         TexCoord             TEXTURE_COORD        tex_coord          t 
         Vertex               VERTEX               position           p
         VertexAttrib         N/A             
        ==================  ==================   ================   =====


        """
        if count is None:
           count = self.indices.size   # this is what the glumpy original does
        pass

        gl.glPushClientAttrib( gl.GL_CLIENT_VERTEX_ARRAY_BIT )
        gl.glBindBuffer( gl.GL_ARRAY_BUFFER, self.vertices_id )
        gl.glBindBuffer( gl.GL_ELEMENT_ARRAY_BUFFER, self.indices_id )

        for attribute in self.generic_attributes:
            attribute.enable()
        for c in self.attributes.keys():
            if c in what:
                self.attributes[c].enable()

        gl.glDrawElements( mode, count, gl.GL_UNSIGNED_INT, ctypes.c_void_p(offset) )

        gl.glBindBuffer( gl.GL_ELEMENT_ARRAY_BUFFER, 0 ) 
        gl.glBindBuffer( gl.GL_ARRAY_BUFFER, 0 ) 
        gl.glPopClientAttrib( )

DRAWMODE = { 'lines':gl.GL_LINES, 'points':gl.GL_POINTS, }
   


 
class DAEPhotons(object):
    """
    A wrapper around the underlying `photons` instance that 
    handles presentation.
    """ 
    def __init__(self, photons, event, pmtid=None ):

        self.invalidate_photons()
        self._photons = photons  
        pass
        self.event = event        # for access to qcut
        config = event.config
        self.config = config
        self.pmtid = pmtid        # unused
        pass
        if not event is None:
            self.reconfig([
                       ['fpholine', config.args.fpholine],
                       ['pholine',  config.args.pholine],
                       ['fphopoint',config.args.fphopoint],
                       ['phopoint', config.args.phopoint],
                      ])

    def __repr__(self):
        return "%s %s " % (self.__class__.__name__, self.nphotons)
    

    nphotons = property(lambda self:len(self._photons))
    vertices = property(lambda self:self._photons.pos)   # allows to be treated like DAEMesh 
    momdir = property(lambda self:self._photons.dir)

    def invalidate_photons(self):
        """
        When changing photons everything is invalidated
        """ 
        self._photons = None
        self._mesh = None
        self._color = None
        self._mode = None
        self._ldata = None   
        self._pdata = None   
        self._drawmode = None
        self._lvbo = None   
        self._pvbo = None   

    def invalidate_vbo(self):
        """
        When reconfiguring presentation just these are invalidated
        """
        self._mode = None
        self._ldata = None   
        self._pdata = None   
        self._drawmode = None
        self._lvbo = None   
        self._pvbo = None   

    def _set_photons(self, photons):
        """
        photons setter invalidates everything
        """
        self.invalidate_photons()
        self._photons = photons
    def _get_photons(self):
        return self._photons    
    photons = property(_get_photons, _set_photons)


    def _get_color(self):
        if self._color is None:
            self._color = self.wavelengths2rgb()
        return self._color
    color = property(_get_color)

    def _get_pdata(self):
        if self._pdata is None:
            self._pdata = self.create_pdata()
        return self._pdata
    pdata = property(_get_pdata)         

    def _get_ldata(self):
        if self._ldata is None:
            self._ldata = self.create_ldata()
        return self._ldata
    ldata = property(_get_ldata)         

    def _get_pvbo(self):
        if self._pvbo is None:
           self._pvbo = self.create_vbo(self.pdata)  
        return self._pvbo
    pvbo = property(_get_pvbo)  

    def _get_lvbo(self):
        if self._lvbo is None:
           self._lvbo = self.create_vbo(self.ldata)  
        return self._lvbo
    lvbo = property(_get_lvbo)  

    def _get_mesh(self):
        if self._mesh is None:
            self._mesh = DAEMesh(self.vertices)
        return self._mesh
    mesh = property(_get_mesh)

    def _get_mode(self):
        if self._mode is None:
            self._mode = 'lines' if self.config.args.pholine else 'points'
        return self._mode
    mode = property(_get_mode)

    def _get_drawmode(self):
        if self._drawmode is None:
            self._drawmode = DRAWMODE[self.mode]
        return self._drawmode
    drawmode = property(_get_drawmode)


    def wavelengths2rgb(self):
        color = np.zeros(self.nphotons, dtype=(np.float32, 4))
        for i,wl in enumerate(self.photons.wavelengths):
            color[i] = wav2RGB(wl)
        return color

    def create_pdata(self):
        data = np.zeros(self.nphotons, [('position', np.float32, 3), 
                                        ('color',    np.float32, 4)]) 
        data['position'] = self.vertices
        data['color']    = self.color
        return data

    def create_ldata(self):
        """
        #. interleave the photon positions with sum of photon position and direction
        #. interleaved double up the colors 
        """
        data = np.zeros(2*self.nphotons, [('position', np.float32, 3), 
                                          ('color',    np.float32, 4)]) 

        vertices = np.empty((len(self.vertices)*2,3), dtype=self.vertices.dtype )
        vertices[0::2] = self.vertices
        vertices[1::2] = self.vertices + self.momdir*self.config.args.fpholine

        colors = np.empty((len(self.color)*2,4), dtype=self.color.dtype )
        colors[0::2] = self.color
        colors[1::2] = self.color

        data['position'] = vertices
        data['color']    = colors
        return data 

    def create_vbo(self, data):
        log.info("create_vbo for %s photons" % self.nphotons)
        indices = np.arange( data.size, dtype=np.uint32)  
        return MyVertexBuffer( data, indices  )

    def draw(self):
        """
        qcut restricts elements drawn, the default of 1 corresponds to all
        """ 
        qcount = int(len(self.pdata)*self.event.qcut)
        self.lvbo.draw(mode=gl.GL_LINES,  what='pc', count=2*qcount, offset=0 )
        self.pvbo.draw(mode=gl.GL_POINTS, what='pc', count=qcount  , offset=0 )


    def reconfig(self, conf):
        update = False
        for k, v in conf:
            if k == 'fpholine':
                self.config.args.fpholine = v
                update = True
            elif k == 'fphopoint':
                self.config.args.fphopoint = v
                gl.glPointSize(self.config.args.fphopoint)
            elif k == 'pholine':
                self.config.args.pholine = True
                update = True
            elif k == 'phopoint':
                self.config.args.pholine = False
                update = True
            else:
                log.info("ignoring %s %s " % (k,v))
            pass 
        pass
        if update:
            self.invalidate_vbo()

    def draw_extremely_slowly(self):
        """
        """
        gl.glDisable( gl.GL_LIGHTING )
        gl.glDisable( gl.GL_DEPTH_TEST )

        for i in range(self.nphotons):
            a = self.vertices[i]
            b = a + self.momdir[i]*100

            gl.glBegin( gl.GL_LINES)
            gl.glVertex3f( *a )
            gl.glVertex3f( *b )
            gl.glEnd()
        pass 

        gl.glEnable( gl.GL_LIGHTING )
        gl.glEnable( gl.GL_DEPTH_TEST )




if __name__ == '__main__':
    pass




