#!/usr/bin/env python
"""

"""
import logging
import numpy as np
log = logging.getLogger(__name__)

from env.graphics.color.wav2RGB import wav2RGB
try:
    from chroma.event import Photons
except ImportError:
    from photons import Photons

from daegeometry import DAEMesh 
from daephotonsparam import DAEPhotonsParam
from daephotonsmenuctrl import DAEPhotonsMenuController
from daephotonsrenderer import DAEPhotonsRenderer
# NB no OpenGL imports allowed here  

class DAEPhotons(object):
    """
    Coordinator class handling photon presentation, 
    specifics belong in constituents

    #. `_photons` CPL instance
    #. `param` presentation parameters
    #. `menuctrl` GLUT menus  
    #. `renderer` OpenGL drawing

    """ 
    def __init__(self, photons, event ):
        """
        :param photons:
        """ 
        self.event = event       
        self.legacy = event.config.args.legacy
        log.info("DAEPhotons legacy %s " % self.legacy )
        self.invalidate_photons()
        self._photons = photons  

        self.param = DAEPhotonsParam( event.config)
        self.menuctrl = DAEPhotonsMenuController( event.config.rmenu, self.param )
        self.renderer = DAEPhotonsRenderer(self, event.scene.chroma) # pass chroma context to renderer for PyCUDA/OpenGL interop tasks 

    def draw(self):
        if not self._photons is None:
            self.renderer.draw()

    def handle_updated_photons(self):
        if not self._photons is None:
            self.menuctrl.update( self._photons )    

    def __repr__(self):
        return "%s %s " % (self.__class__.__name__, self.nphotons)


    position     = property(lambda self:self._photons.pos if not self._photons is None else None)   
    direction    = property(lambda self:self._photons.dir if not self._photons is None else None)
    wavelength   = property(lambda self:self._photons.wavelengths if not self._photons is None else None)
    weight       = property(lambda self:self._photons.weights if not self._photons is None else None)
    flags        = property(lambda self:self._photons.flags if not self._photons is None else None)
    polarization = property(lambda self:self._photons.pol if not self._photons is None else None)
    time         = property(lambda self:self._photons.t if not self._photons is None else None)
    last_hit_triangle = property(lambda self:self._photons.last_hit_triangles if not self._photons is None else None)

    nphotons     = property(lambda self:len(self._photons) if not self._photons is None else 0)
    vertices     = property(lambda self:self.position)  # allows to be treated like DAEMesh 

    def reconfig(self, conf):
        update = self.param.reconfig(conf)
        if update:
            self.invalidate_buffer()

    def invalidate_photons(self):
        """
        When changing photons everything is invalidated
        """ 
        self._photons = None

        self._indices = None
        self._mesh = None
        self._color = None

        self._ldata = None   
        self._lbuffer = None   
        self._lindices = None   

        self._pdata = None   
        self._pbuffer = None   
        self._pindices = None   

    def invalidate_buffer(self):
        """
        When reconfiguring presentation just these are invalidated
        """
        log.info("invalidate_buffer")
        self._ldata = None   
        self._lbuffer = None   
        self._lindices = None   

        self._pdata = None   
        self._pbuffer = None   
        self._pindices = None   

        self._indices = None

    def _set_photons(self, photons):
        """
        photons setter invalidates everything
        """
        self.invalidate_photons()
        self._photons = photons
        self.handle_updated_photons()

    def _get_photons(self):
        return self._photons    
    photons = property(_get_photons, _set_photons)


    def _get_color(self):
        if self._color is None:
            self._color = self.wavelengths2rgb()
        return self._color
    color = property(_get_color)



    def _get_ldata(self):
        if self._ldata is None:
            self._ldata = self.create_ldata()
        return self._ldata
    ldata = property(_get_ldata)         

    def _get_lindices(self):
        if self._lindices is None:
            self._lindices = self.create_vindices(doubled=True)
        return self._lindices
    lindices = property(_get_lindices, doc="doubled up indices needed for line drawing")         

    def _get_lbuffer(self):
        if self._lbuffer is None:
           self._lbuffer = self.renderer.create_buffer(self.ldata, self.lindices)  
        return self._lbuffer
    lbuffer = property(_get_lbuffer)  



    def _get_pdata(self):
        if self._pdata is None:
            if self.legacy:
                self._pdata = self.create_pdata_legacy()
            else:
                self._pdata = self.create_pdata()
            pass 
        return self._pdata
    pdata = property(_get_pdata)         

    def _get_pindices(self):
        if self._pindices is None:
            self._pindices = self.create_vindices(doubled=False)
        return self._pindices
    pindices = property(_get_pindices, doc="non-doubled up indices used for point drawing")         

    def _get_pbuffer(self):
        if self._pbuffer is None:
           self._pbuffer = self.renderer.create_buffer(self.pdata, self.pindices, self.position_name )  
        return self._pbuffer
    pbuffer = property(_get_pbuffer)  





    def _get_mesh(self):
        if self._mesh is None:
            self._mesh = DAEMesh(self._photons.pos)
        return self._mesh
    mesh = property(_get_mesh)

    def _get_indices(self):
        if self._indices is None:
            self._indices = self.indices_selection()
        return self._indices 
    indices = property(_get_indices)

    def wavelengths2rgb(self):
        if self.nphotons == 0:return None
        color = np.zeros(self.nphotons, dtype=(np.float32, 4))
        for i,wl in enumerate(self.wavelength):
            color[i] = wav2RGB(wl)
        return color

    def _get_pnumfields(self):
        return len(self.pdata.dtype.fields) if not self.pdata is None else None
    pnumfields = property(_get_pnumfields)


    # using 'position' uses traditional glVertexPointer furnishing gl_Vertex to shader
    # using smth else eg 'vposition' use generic attribute , which requires force_attribute_zero for anythinh to appear
 
    position_name = property(lambda self:"position_weight") 
    num4vec = property(lambda self:"5")
    debug_shader = property(lambda self:False)  # when True switch off geometry shader for debugging 

    def create_pdata(self):
        """
        Simplify access from CUDA kernels by adopting 4*quads::

            struct VPhoton { 
               float4 position_weight,                 # 4*4 = 16  
               float4 direction_wavelength,            # 4*4 = 16     
               float4 polarization_time,               # 4*4 = 16  48
               uint4  flags                            # 4*4 = 16  64    
               int4   last_hit_triangle                # 4*4 = 16  80    

        """
        if self.nphotons == 0:return None

        dtype = np.dtype([ 
            ('position_weight'   ,        np.float32, 4 ), 
            ('direction_wavelength',      np.float32, 4 ), 
            ('polarization_time',         np.float32, 4 ), 
            ('flags',                     np.uint32,  4 ), 
            ('last_hit_triangle',         np.int32,   4 ), 
          ])

        nvert = self.nphotons
        data = np.zeros(nvert, dtype )

        def pack31_( name, a, b ):
            data[name][:,:3] = a
            data[name][:,3] = b

        pack31_( 'position_weight',      self.position ,    self.weight )
        pack31_( 'direction_wavelength', self.direction,    self.wavelength )
        pack31_( 'polarization_time',    self.polarization, self.time  )

        def pack1111_( name, a, b, c, d ):
            data[name][:,0] = a
            data[name][:,1] = b
            data[name][:,2] = c
            data[name][:,3] = d

        ones_ = lambda _:np.ones( nvert, dtype=_ )  
        pack1111_('flags',             self.flags,             ones_(np.uint32), ones_(np.uint32), ones_(np.uint32) )
        pack1111_('last_hit_triangle', self.last_hit_triangle, ones_(np.int32),  ones_(np.int32), ones_(np.int32) )

        return data


    def create_pdata_legacy(self):
        """
        Create numpy structured array (effectively an array of structs).

        #. just photon positions and colors, for the points
        #. very regular dtype for on device float4 convenience

        """
        if self.nphotons == 0:return None
       
        nvert = self.nphotons
        dtype = np.dtype([ 
            ('position',  np.float32, 4 ), 
            ('direction', np.float32, 4 ), 
            ('color',     np.float32, 4 ), 
          ])

        data = np.zeros(nvert, dtype )
        data['position'][:,:3] = self.vertices
        data['position'][:,3] = np.ones( nvert, dtype=np.float32 )  # fill in the w with ones

        data['direction'][:,:3] = self.direction*self.param.fpholine
        data['direction'][:,3]  = self.wavelength     # stuff the wavelength into 4th slot of momdir

        data['color'] = self.color

        return data

    def create_ldata(self):
        """
        For line presentation (without using geometry shaders)
        have to double up the vertices and colors:

        #. interleave the photon positions with sum of photon position and direction
        #. interleaved double up the colors 

        """
        if self.nphotons == 0:return None
        data = np.zeros(2*self.nphotons, [('position', np.float32, 3), 
                                          ('color',    np.float32, 4)]) 

        assert len(self.vertices) == len(self.color)
        nvert = len(self.vertices)*2
        vertices = np.empty((nvert,3), dtype=self.vertices.dtype )
        vertices[0::2] = self.vertices
        vertices[1::2] = self.vertices + self.momdir*self.param.fpholine

        colors = np.empty((nvert,4), dtype=self.color.dtype )
        colors[0::2] = self.color
        colors[1::2] = self.color

        data['position'] = vertices
        data['color']    = colors
        return data

    def _get_qcount(self):
        return int(self.nphotons*self.event.qcut)
    qcount = property(_get_qcount, doc="Photon count modulated by qcut which varies between 0 and 1. Used for partial drawing based on a sorted quantity." ) 

    def indices_selection(self):
        """
        Use the photon history flags (array of bit fields)
        to make bit mask and history selections
        """
        if self.nphotons == 0:return None
        npho = self.nphotons
        flags = self.photons.flags   
        nflag = len(flags)
        assert nflag == npho, (nflg, npho)

        self.menuctrl.update_history_menu( self.photons )    # dislike positioning, "side effect"

        mask = self.param.mask 
        bits = self.param.bits 

        if mask is None and bits is None:
            indices = np.arange( npho, dtype=np.uint32)  
        elif not mask is None:
            indices = np.where( flags & mask )[0]    # bitwise AND for flags selection
        elif not bits is None:
            indices = np.where( flags == bits )[0]   # equality for history bits selection
        else:
            assert 0

        log.info("indices_selection : %s " % len(indices) )

        if len(indices) == 0:
            log.warn("added token single indice, to avoid empty indices ")
            indices = np.arange( 1, dtype=np.uint32) 
 
        return indices



    def create_vindices(self, doubled=False):
        """
        #. when no photons are selected, a token single indice is used that avoids a crash
        #. for non-doubled vindices (originally used for points drawing) its 1-1 between vertices and photons
        #. for doubled vindices (originally used for lines drawing) its 2-1 between vertices and photons
        """
        if self.nphotons == 0:return None
        indices = self.indices
        if len(indices) == 0:
            vindices = np.arange( 1, dtype=np.uint32 ) 
        else: 
            if doubled:
                vindices = np.empty(2*len(indices), dtype=np.uint32)
                vindices[0::2] = 2*indices
                vindices[1::2] = 2*indices + 1
            else:
                vindices = indices
        pass
        log.info("create_vindices %s " % len(vindices))
        return vindices



if __name__ == '__main__':
    pass




