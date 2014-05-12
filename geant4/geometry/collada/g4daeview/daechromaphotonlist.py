#!/usr/bin/env python
"""
For checking using LXe example::

    g4daeview.sh -p lxe -g 1: -n1

    lxe-
    lxe-test   # sling some photons via RootZMQ


Interactive probing::

   (chroma_env)delta:g4daeview blyth$ ipython daechromaphotonlist.py -i

    In [12]: ddir = map(lambda _:np.linalg.norm(_), cpl.dir )

    In [14]: print ddir
    [1.4209839, 1.0561398, 1.1840367, 0.94664037, 1.0857934, 1.4024338, 1.2497476, 0.96031791, 0.92489851, 1.2788764, 1.2204158, 1.2413212, 0.33396932, 1.0090759, 1.3084445, 0.96862358, 1.1485049, 0.8632732, 1.1312609, 0.65599948, 1.1350429, 0.64752048, 0.83116335, 1.3650827, 0.11439443, 0.58249426, 1.2186502, 1.1817784, 1.1500614, 0.88921714, 0.78321654, 0.55772144, 0.97450805, 0.5246563, 0.64278752, 0.17736676, 1.0694224, 0.93206048, 0.9899205, 1.0367258, 1.1141851, 1.1176836, 1.3933165, 0.76192909, 1.2836145, 1.1873106, 0.93039864, 0.98263144, 1.1388388, 0.65394604, 0.73478901, 1.0558159, 0.77594322, 0.87060881, 0.78373843, 1.2936558, 1.2129512, 1.2595264, 1.2817491, 0.99562281, 1.2855144, 1.1620266, 1.1792105, 1.3162464, 1.4996835, 0.9008953, 1.0214221, 1.0381697, 1.2537543, 1.0436105, 0.89909387, 0.85806465, 1.2673098, 0.81995475, 1.24819, 1.2298234, 0.81444257, 1.2929296, 1.3538636, 1.2676951, 0.91203678, 1.1772467, 1.191206, 0.95119107, 1.1960018, 0.49314541, 0.92237413, 0.70844412, 1.3627962, 1.1306297, 1.0215813, 0.67949617, 0.94199985, 0.77324378, 1.3687329, 0.98732764, 1.1357194, 0.88006437, 0.6298191, 1.1748897]





"""
import logging
log = logging.getLogger(__name__)


import glumpy as gp
import numpy as np
import OpenGL.GL as gl
import OpenGL.GLUT as glut

xyz_ = lambda x,y,z,dtype:np.column_stack((np.array(x,dtype=dtype),np.array(y,dtype=dtype),np.array(z,dtype=dtype))) 




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

        self.vbo = gp.graphics.VertexBuffer( self.data, self.indices  )

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
        self.vbo.draw(mode=gl.GL_POINTS, what='p' )

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





