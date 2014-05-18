#!/usr/bin/env python
"""
For checking using LXe example::

    g4daeview.sh -p lxe -g 1: -n1

    lxe-
    lxe-test   # sling some photons via RootZMQ

Interactive probing::

   (chroma_env)delta:g4daeview blyth$ ipython daechromaphotonlist.py -i

    In [12]: ddir = map(lambda _:np.linalg.norm(_), cpl.dir )


Classes `chroma.event.Photons` and `photons.Photons` 
should be identical, doing fallback import as want to support installations
without chroma

TODO: test without chroma running


NO_HIT                                     1   0x1      
BULK_ABSORB                                2   0x2      
SURFACE_DETECT                             4   0x4      
SURFACE_ABSORB                             8   0x8      
RAYLEIGH_SCATTER                          16   0x10      
REFLECT_DIFFUSE                           32   0x20      
REFLECT_SPECULAR                          64   0x40      
SURFACE_REEMIT                           128   0x80      
SURFACE_TRANSMIT                         256   0x100      
BULK_REEMIT                              512   0x200      
NAN_ABORT                         2147483648   0x80000000      


"""
import logging
import numpy as np
log = logging.getLogger(__name__)


def count_unique(vals):
    """
    http://stackoverflow.com/questions/10741346/numpy-frequency-counts-for-unique-values-in-an-array
    """
    uniq = np.unique(vals)
    bins = uniq.searchsorted(vals)
    return np.vstack((uniq, np.bincount(bins))).T


import ctypes
import glumpy as gp
import OpenGL.GL as gl
import OpenGL.GLUT as glut


DRAWMODE = { 'lines':gl.GL_LINES, 'points':gl.GL_POINTS, }

from env.graphics.color.wav2RGB import wav2RGB

try:
    from chroma.event import Photons, mask2arg_, arg2mask_, PHOTON_FLAGS
except ImportError:
    from photons import Photons, mask2arg_, arg2mask_, PHOTON_FLAGS


from daegeometry import DAEMesh 
from daemenu import DAEMenu


def arg2mask( argl ):
    """ 
    Return strings representing integers as integers
    otherwise perform enum name to mask conversion.
    """
    if argl is None or argl == "NONE":return None
    mask = 0 
    try:
        mask = int(argl)
    except ValueError:
        mask = arg2mask_(argl)
    pass
    return mask


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
        pass
        self.config.rmenu.addSubMenu(self.make_submenu()) # RIGHT menu hookup

    def __repr__(self):
        return "%s %s " % (self.__class__.__name__, self.nphotons)
    

    nphotons = property(lambda self:len(self._photons))
    vertices = property(lambda self:self._photons.pos)   # allows to be treated like DAEMesh 
    momdir = property(lambda self:self._photons.dir)


    def make_submenu(self):
        log.info("make_submenu")
        photons = DAEMenu("photons")

        flags = DAEMenu("flags")
        flags.add("ANY", self.flags_callback )
        for name in sorted(PHOTON_FLAGS, key=lambda _:PHOTON_FLAGS[_]):
            flags.add(name, self.flags_callback )
        pass
        photons.addSubMenu(flags)

        history = DAEMenu("history")
        self.history = history
        photons.addSubMenu(history)    # a placeholder menu to be changed once propagation stepping is done
        return photons

    def change_history_menu(self, flags):
        """
        TODO: avoid this being unecessarily called twice, for each create_vbo
        """
        pass
        nflg = len(flags)
        history = count_unique(flags)
        log.info("unique flag combinations %s " % len(history))
        #print "\n".join(["[0x%-10x] %5d (%5.2f): %s " % (_[0],_[1],float(_[1])/nflg, mask2arg_(_[0])) for _ in sorted(history,key=lambda _:_[1])])
            
        self.history.addnew( "ANY", self.history_callback, mask=None )

        for mask,count in sorted(history,key=lambda _:_[1], reverse=True):
            frac = float(count)/nflg
            title = "[0x%x] %d (%5.2f): %s " % (mask,count,frac,mask2arg_(mask)) 
            self.history.addnew( title, self.history_callback, mask=mask )
        pass
        self.history.replace_menu_items()
 
    def history_callback(self, item):
        self.config.args.mask = None
        self.config.args.bits = item.extra['mask']  
        self.invalidate_vbo()
        self.config.rmenu.dispatch('on_needs_redraw')

    def flags_callback(self, item ):
        name = item.title
        allowed = PHOTON_FLAGS.keys() + ['ANY']
        assert name in allowed, name
        log.info("flags_callback setting config.args.mask to %s " % name )
        if name == 'ANY':name = 'NONE'
        self.config.args.mask = name 
        self.config.args.bits = None 
        self.invalidate_vbo()
        self.config.rmenu.dispatch('on_needs_redraw')

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
        log.info("invalidate_vbo")
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
        """
        """

        nvtx = data.size
        npho = self.nphotons
        flags = self.photons.flags   # large array of photon bit fields
        nflg = len(flags)

        self.change_history_menu( flags )

        argm = self.config.args.mask 
        argb = self.config.args.bits 


        mask = arg2mask(argm)
        bits = arg2mask(argb)

        log.info("argm %s mask %s " % (argm, mask))
        log.info("argb %s bits %s " % (argm, bits))

        log.info("create_vbo npho %s nvtx %s nflg %s mask %s " % (npho,nvtx,nflg,mask))

        assert nflg == npho, (nflg, npho)
        assert nvtx == npho or nvtx == 2*npho, (nvtx,npho)

        if mask is None and bits is None:
            pindices = np.arange( nvtx, dtype=np.uint32)  
        elif not mask is None:
            pindices = np.where( flags & mask )[0]    # bitwise AND for flags selection
        elif not bits is None:
            pindices = np.where( flags == bits )[0]   # equality for history bits selection
        else:
            assert 0

        log.info("photon selection : %s " % len(pindices) )
 
        if len(pindices) == 0:
            vindices = np.arange( 1, dtype=np.uint32 ) # token 1 indice when mask kills all
        else:
            # for points pvbo its 1-1 between vertices and photons, wheras for lines lvbo is 2-1
            if nvtx == npho:
                vindices = pindices
            elif nvtx == npho*2:
                vindices = np.empty(2*len(pindices), dtype=np.uint32)
                vindices[0::2] = 2*pindices
                vindices[1::2] = 2*pindices + 1
                log.info(" lvbo vindices %s " % len(vindices))
            else:
                assert 0, (nvtx,npho)
            pass
        pass
        return MyVertexBuffer( data, vindices  )

    def draw(self):
        """
        qcut restricts elements drawn, the default of 1 corresponds to all

        Note that the VBO vertices are duplicated once for the line and once for
        the points, presumably there is some clever way to control the strides to
        avoid that ?

        Changing presentation must account for the OpenGL state when this
        gets called from daeframeghandler
        """ 
        qcount = int(len(self.pdata)*self.event.qcut)

        #gl.glDisable( gl.GL_LIGHTING )
        # 
        #if self.event.scene.photonmagic: 
        #    gl.glDisable( gl.GL_DEPTH_TEST )
        #else:
        #    gl.glEnable( gl.GL_DEPTH_TEST )

        self.lvbo.draw(mode=gl.GL_LINES,  what='pc', count=2*qcount, offset=0 )
        self.pvbo.draw(mode=gl.GL_POINTS, what='pc', count=qcount  , offset=0 )

        #gl.glEnable( gl.GL_LIGHTING )
        #gl.glEnable( gl.GL_DEPTH_TEST )



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
            elif k == 'mask':
                self.config.args.mask = v
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




