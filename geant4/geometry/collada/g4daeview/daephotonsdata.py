#!/usr/bin/env python
"""

"""
import logging
import numpy as np
log = logging.getLogger(__name__)

from env.graphics.color.wav2RGB import wav2RGB

# gl/glumpy imports not allowed, this is for pure numpy data manipulations

class DAEPhotonsDataBase(object):
    """
    Responsible for "casting" the storage/transport oriented `ChromaPhotonList` 
    instances into the form needed to construct Vertex Buffer Objects.
    """
    def __init__(self, photons, param ):
        """
        :param photons: `chroma.event.Photons` instance (`photons_fallback.Photons` when no chroma) 
        """
        self.invalidate()
        self._photons = photons  
        self._param = param 

    nphotons     = property(lambda self:len(self._photons) if not self._photons is None else 0)
    position     = property(lambda self:self._photons.pos if not self._photons is None else None)   
    direction    = property(lambda self:self._photons.dir if not self._photons is None else None)
    wavelength   = property(lambda self:self._photons.wavelengths if not self._photons is None else None)
    weight       = property(lambda self:self._photons.weights if not self._photons is None else None)
    flags        = property(lambda self:self._photons.flags if not self._photons is None else None)
    polarization = property(lambda self:self._photons.pol if not self._photons is None else None)
    time         = property(lambda self:self._photons.t if not self._photons is None else None)
    last_hit_triangle = property(lambda self:self._photons.last_hit_triangles if not self._photons is None else None)

    def invalidate(self):
        """
        When changing photons everything is invalidated
        """ 
        self._photons = None
        self._data = None   
        self._indices = None   
        self._color = None

    def _get_photons(self):
        return self._photons    
    def _set_photons(self, photons):
        self.invalidate()
        self._photons = photons
    photons = property(_get_photons, _set_photons)

    def _get_param(self):
        return self._param
    def _set_param(self, param):
        self.invalidate()   # huh is this invalidation still needed ?
        self._param = param
    param = property(_get_param, _set_param)


    def _get_data(self):
        if self._data is None:
            self._data = self.create_data()
        return self._data
    data = property(_get_data)         

    def _get_vindices(self):
        if self._vindices is None:
            self._vindices = self.create_vindices(doubled=True)
        return self._vindices
    vindices = property(_get_vindices, doc="")         

    def _get_indices(self):
        if self._indices is None:
            self._indices = np.arange( self.nphotons, dtype=np.uint32)  
        return self._indices 
    indices = property(_get_indices)

    def indices_selection(self):
        """
        MOVING THIS TO CUDA KERNEL

        Use the photon history flags (array of bit fields)
        to make bit mask and history selections
        """
        if self.nphotons == 0:return None

        npho = self.nphotons
        flags = self.flags   
        nflag = len(flags)

        assert nflag == npho, (nflg, npho)

        mask = self.param.mask 
        bits = self.param.bits 

        if mask is -1 and bits is -1:
            indices = np.arange( npho, dtype=np.uint32)  
        elif not mask is -1:
            indices = np.where( flags & mask )[0]    # bitwise AND for flags selection
        elif not bits is -1:
            indices = np.where( flags == bits )[0]   # equality for history bits selection
        else:
            assert 0

        if len(indices) == 0:
            log.warn("added token single indice, to avoid empty indices ")
            log.info("indices_selection npho %s nflag %s " % ( npho, nflag ))
            log.info("indices_selection : %s mask %s bits %s  " % ( len(indices), mask, bits) )
            indices = np.arange( 1, dtype=np.uint32) 
 
        return indices



    def wavelengths2rgb(self):
        if self.nphotons == 0:return None
        color = np.zeros(self.nphotons, dtype=(np.float32, 4))
        for i,wl in enumerate(self.wavelength):
            color[i] = wav2RGB(wl)
        return color

    def _get_color(self):
        if self._color is None:
            self._color = self.wavelengths2rgb()
        return self._color
    color = property(_get_color)





class DAEPhotonsData(DAEPhotonsDataBase):
    numquad = 6
    force_attribute_zero = "position_weight"
    def __init__(self, photons, param ):
        DAEPhotonsDataBase.__init__(self, photons, param )
 
    def create_data(self):
        """
        Simplify access from CUDA kernels by adopting 4*quads::

            struct VPhoton { 
               float4 position_weight,                 # 4*4 = 16  
               float4 direction_wavelength,            # 4*4 = 16     
               float4 polarization_time,               # 4*4 = 16  48
               uint4  flags                            # 4*4 = 16  64    
               int4   last_hit_triangle                # 4*4 = 16  80    

          
        #. using 'position' would use traditional glVertexPointer furnishing gl_Vertex to shader
        #. using smth else eg 'position_weight' uses generic attribute , 
           which requires force_attribute_zero for anythinh to appear


        Fake attribute testing

        #. replace self.time, self.flags and self.last_hit_triangle 
           to check getting different types (especially integer) 
           attributes into shaders 

        r012_ = lambda _:np.random.randint(3, size=nvert ).astype(_)  # 0,1,2 randomly 
        r124_ = lambda _:np.array([1,2,4],dtype=_)[np.random.randint(3, size=nvert )]

        fake_time = ones_(np.float32)
        fake_flags = r012_(np.uint32)
        fake_last_hit_triangle = r012_(np.int32)

        #max_uint32 = (1 << 32) - 1 
        #max_int32 =  (1 << 31) - 1 
        #fake_flags = np.tile( 1  , nvert ).astype(np.uint32)
        #fake_last_hit_triangle = np.tile( 1, nvert ).astype(np.int32)

        """
        if self.nphotons == 0:return None

        dtype = np.dtype([ 
            ('position_weight'   ,        np.float32, 4 ), 
            ('direction_wavelength',      np.float32, 4 ), 
            ('polarization_time',         np.float32, 4 ), 
            ('ccolor',                    np.float32, 4 ), 
            ('flags',                     np.uint32,  4 ), 
            ('last_hit_triangle',         np.int32,   4 ), 
          ])

        nvert = self.nphotons
        data = np.zeros(nvert, dtype )

        ones_ = lambda _:np.ones( nvert, dtype=_ )  
        def pack31_( name, a, b ):
            data[name][:,:3] = a
            data[name][:,3] = b

        pack31_( 'position_weight',      self.position ,    self.weight )
        pack31_( 'direction_wavelength', self.direction,    self.wavelength )
        pack31_( 'polarization_time',    self.polarization, self.time  )
        
        data['ccolor'] = np.tile( [1.,0.,0,1.], (nvert,1)).astype(np.float32)    # initialize to red, reset by CUDA kernel

        def pack1111_( name, a, b, c, d ):
            data[name][:,0] = a
            data[name][:,1] = b
            data[name][:,2] = c
            data[name][:,3] = d

        pack1111_('flags',             self.flags,             ones_(np.uint32), ones_(np.uint32), ones_(np.uint32) )
        pack1111_('last_hit_triangle', self.last_hit_triangle, ones_(np.int32),  ones_(np.int32), ones_(np.int32) )
        return data






class DAEPhotonsDataLegacy(DAEPhotonsDataBase):
    numquad = 3
    force_attribute_zero = "position"
    def __init__(self, photons, param ):
        DAEPhotonsDataBase.__init__(self, photons, param )

    def create_data(self):
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
        data['position'][:,:3] = self.position
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

        assert len(self.position) == len(self.color)
        nvert = len(self.position)*2
        position = np.empty((nvert,3), dtype=self.position.dtype )
        position[0::2] = self.position
        position[1::2] = self.position + self.direction*self.param.fpholine

        colors = np.empty((nvert,4), dtype=self.color.dtype )
        colors[0::2] = self.color
        colors[1::2] = self.color

        data['position'] = position
        data['color']    = colors
        return data


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
    # run with ./daephotonsdata.sh for env setup

    from daeconfig import DAEConfig

    config = DAEConfig()
    config.init_parse()
    cpl = config.load_cpl("1")

    from photons import Photons
    photons = Photons.from_cpl(cpl, extend=True)
    print photons

    from daephotonsparam import DAEPhotonsParam
    param = DAEPhotonsParam(config)
    pd = DAEPhotonsData( photons, param )
    data = pd.data

    print pd
    print data
    print data.dtype
   



