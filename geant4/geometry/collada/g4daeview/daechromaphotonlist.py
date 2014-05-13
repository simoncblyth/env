#!/usr/bin/env python
"""
For checking using LXe example::

    g4daeview.sh -p lxe -g 1: -n1

    lxe-
    lxe-test   # sling some photons via RootZMQ


Interactive probing::

   (chroma_env)delta:g4daeview blyth$ ipython daechromaphotonlist.py -i

    In [12]: ddir = map(lambda _:np.linalg.norm(_), cpl.dir )

"""
import logging, ctypes
log = logging.getLogger(__name__)


import glumpy as gp
import numpy as np
import OpenGL.GL as gl
import OpenGL.GLUT as glut

xyz_ = lambda x,y,z,dtype:np.column_stack((np.array(x,dtype=dtype),np.array(y,dtype=dtype),np.array(z,dtype=dtype))) 

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

    

class DAEChromaPhotonList(object):
    def __init__(self, cpl):
        self.cpl = cpl
        self.nphotons = cpl.x.size()
        cpl.Print()
        self.copy_from_cpl(cpl)
        self.create_vbo()

    def copy_from_cpl(self, cpl):
        self.pos = xyz_(cpl.x,cpl.y,cpl.z,np.float32)
        self.dir = xyz_(cpl.px,cpl.py,cpl.pz,np.float32)
        self.pol = xyz_(cpl.polx,cpl.poly,cpl.polz,np.float32)
        self.wavelength = np.array(cpl.wavelength, dtype=np.float32)
        self.t = np.array(cpl.t, dtype=np.float32)
        self.pmtid = np.array(cpl.pmtid, dtype=np.int32)

    def create_vbo(self):
        """
        For time sliding try enhancing glumpy VBO
        to support: glDrawRangeElements

        Then by sorting photons by time at VBO creation and recording 
        appropriate offsets can make an interactive time cut.

        """
        log.info("create_vbo")
        data = np.zeros( self.nphotons, [('position', np.float32, 3)])
        data['position'] = self.pos

        self.data = data
        self.indices = np.arange(data.size,dtype=np.uint32)  # the default used by VertexBuffer if no were indices given

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

        """ 
        #self.vbo.draw(mode=gl.GL_POINTS, what='p' )
        self.vbo.draw(mode=gl.GL_POINTS, what='p', count=100, offset=4000 )

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

    import ROOT
    ROOT.gSystem.Load("$LOCAL_BASE/env/chroma/ChromaPhotonList/lib/libChromaPhotonList")

    from env.chroma.ChromaPhotonList.cpl import random_cpl
    cpl = DAEChromaPhotonList(random_cpl())

    print cpl.dir





