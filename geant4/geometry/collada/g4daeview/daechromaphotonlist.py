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



from daechromaphotonlistbase import DAEChromaPhotonListBase


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
   
 
class DAEChromaPhotonList(DAEChromaPhotonListBase):
    def __init__(self, cpl, event ):
        self.event = event
        DAEChromaPhotonListBase.__init__(self, cpl, timesort=True )

        self.reconfig([
                       ['fpho',event.config.args.fpho],
                       ['pholine',event.config.args.pholine],
                      ])

    def __repr__(self):
        return "%s %s " % (self.mode, self.fpho)

    def reconfig(self, conf):
        update = False
        for k, v in conf:
            if k == 'fpho':
                self.fpho = v
                update = True
            elif k == 'pholine':
                self.set_pholine(v)
                update = True
            else:
                log.info("DCPL ignoring %s %s " % (k,v))
            pass 
        pass
        if update:
            self.create_vbo()

    def set_pholine(self, pholine):
        self.pholine = pholine
        self.mode = 'lines' if pholine else 'points' 
        self.drawmode = DRAWMODE[self.mode]

    def create_vbo(self):
        """
        #. creates structure of nphoton elements populated with zeros
           the populates the 'position' slot with pos

        #. indices provides element array from 0:nphotons-1
        """
        log.info("create_vbo for %s photons" % self.nphotons)
        self.dirty = False
        if self.mode == 'points':
            data = np.zeros(self.nphotons, [('position', np.float32, 3), 
                                            ('color',    np.float32, 4)]) 
            data['position'] = self.pos
            data['color']    = self.color

        elif self.mode == 'lines': 
            data = np.zeros(2*self.nphotons, [('position', np.float32, 3), 
                                              ('color',    np.float32, 4)]) 

            # interleave the photon positions with sum of photon position and direction
            vertices = np.empty((len(self.pos)*2,3), dtype=self.pos.dtype )
            vertices[0::2] = self.pos
            vertices[1::2] = self.pos + self.dir*self.fpho

            # interleaved double up the colors 
            colors = np.empty((len(self.color)*2,4), dtype=self.color.dtype )
            colors[0::2] = self.color
            colors[1::2] = self.color

            data['position'] = vertices
            data['color']    = colors

        else:
            assert 0


        self.data = data
        self.indices = np.arange( data.size, dtype=np.uint32)  
        self.vbo = MyVertexBuffer( self.data, self.indices  )

    def draw(self):
        """
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


        The qcut cuts away at the elements to be presented,     
        the default of 1 corresponds to all


        """ 

        qcut = self.event.qcut
        tot = len(self.data)

        # initially tried changing both count and offset, 
        # it works but gives SIGABRTs after a short while
        # just changing the count has not given trouble yet 
        #
        qoffset, qcount = 0, int(tot*qcut)
        assert qcount <= tot

        #log.info("tot %d qcut %s qcount %d qoffset %d " % (tot, qcut, qcount, qoffset ))  

      
        self.vbo.draw(mode=self.drawmode, what='pc', count=qcount, offset=qoffset )

    def draw_slowly(self):
        """
        TODO: 

        #. assign line color based on wavelength, see `specrend-`
        #. howabout pol/t/pmtid ?

        #. investigate using VBO or sprite based particle systems for photon "visualization"

        """
        gl.glDisable( gl.GL_LIGHTING )
        gl.glDisable( gl.GL_DEPTH_TEST )

        for i in range(self.nphotons):
            a = self.pos[i]
            b = a + self.dir[i]*100

            gl.glBegin( gl.GL_LINES)
            gl.glVertex3f( *a )
            gl.glVertex3f( *b )
            gl.glEnd()
        pass 

        gl.glEnable( gl.GL_LIGHTING )
        gl.glEnable( gl.GL_DEPTH_TEST )




if __name__ == '__main__':
    pass




