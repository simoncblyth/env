#!/usr/bin/env python
"""
For standalone testing use::

    ./daephotonsdata.sh --prescale 10


.. warning:: gl/glumpy imports **NOT ALLOWED** this is for pure numpy data manipulations

"""
import logging
import numpy as np
log = logging.getLogger(__name__)

from env.graphics.color.wav2RGB import wav2RGB

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

    prescale = property(lambda self:self._param.prescale)
    max_slots = property(lambda self:self._param.max_slots)

    def __repr__(self):
        return "\n".join([
            "%s ntruephotons %s nphotons %s prescale %s " % ( self.__class__.__name__, self.ntruephotons, self.nphotons, self.prescale ),
            str(self.data.dtype),
            str(self.data),
            "nbytes: %s size:%s itemsize:%s size*itemsize:%s" % ( self.data.nbytes, self.data.size, self.data.itemsize, self.data.size*self.data.itemsize )
            ])

    ntruephotons = property(lambda self:len(self._photons) if not self._photons is None else 0, doc="Original number of photons from CPL")
    is_prescaled = property(lambda self:not self.prescale in (1, None))

    def _get_nphotons(self):
        """
        Number of photons, with prescale division when prescale is applied
        """ 
        return self.ntruephotons//self.prescale if self.is_prescaled else self.ntruephotons
    nphotons     = property(_get_nphotons)

    def create_sample_indices(self):
        """
        Original (unscaled) photon array indices, used to 
        construct prescaled arrays when prescale > 1
        """
        ntruephotons = self.ntruephotons 
        nphotons = self.nphotons
        if ntruephotons == nphotons:
            return np.arange( ntruephotons, dtype=np.uint32)  
        else:
            return np.linspace(0, ntruephotons, num=nphotons).astype('uint8') 

    def _get_sample_indices(self):
        """
        List of indices to be applied to true arrays to possibly 
        create prescaled arrays.
        """ 
        if self._photons is None:
            return None
        if self._sample_indices is None:
            self._sample_indices = self.create_sample_indices()
        return self._sample_indices 
    sample_indices = property(_get_sample_indices, doc=_get_sample_indices.__doc__)

    namemap = {'position':'pos',
               'direction':'dir',
               'wavelength':'wavelengths',
               'weight':'weights',
               'flags':'flags',
               'polarization':'pol',
               'time':'t',
               'last_hit_triangle':'last_hit_triangles',
              }

    def _get_array(self, name):
        """
        :param name: 
        :return: photon property array, potentially prescaled
        """
        if self._photons is None:
            return None
        a = getattr(self._photons, self.namemap[name])
        return a[self.sample_indices] if self.is_prescaled else a

    position     = property(lambda self:self._get_array('position'))   
    direction    = property(lambda self:self._get_array('direction'))   
    wavelength   = property(lambda self:self._get_array('wavelength'))   
    weight       = property(lambda self:self._get_array('weight'))   
    flags        = property(lambda self:self._get_array('flags'))   
    polarization = property(lambda self:self._get_array('polarization'))   
    time         = property(lambda self:self._get_array('time'))   
    last_hit_triangle = property(lambda self:self._get_array('last_hit_triangle'))   


    def invalidate(self):
        """
        When changing photons everything is invalidated
        """ 
        log.info("invalidate")
        self._photons = None
        self._data = None   
        self._indices = None   
        self._color = None
        self._ccolor = None
        self._sample_indices = None

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
        """
        List of indices, possibly within the post-prescaled array.
        """ 
        if self._indices is None:
            self._indices = np.arange( self.nphotons, dtype=np.uint32)  
        return self._indices 
    indices = property(_get_indices, doc=_get_indices.__doc__)

    def indices_selection(self):
        """
        NOW DONE IN CUDA KERNEL

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
        self._ccolor = None

    nvert = property(lambda self:self.nphotons*self.max_slots)
 
    def create_data(self):
        """
        :return: numpy named constituent array with numquad*quads structure 

        #. using 'position' would use traditional glVertexPointer furnishing gl_Vertex to shader
        #. using smth else eg 'position_weight' uses generic attribute , 
           which requires force_attribute_zero for anythinh to appear

        Fake attribute testing

        #. replace self.time, self.flags and self.last_hit_triangle 
           to check getting different types (especially integer) 
           attributes into shaders 

        r012_ = lambda _:np.random.randint(3, size=nvert ).astype(_)  # 0,1,2 randomly 
        r124_ = lambda _:np.array([1,2,4],dtype=_)[np.random.randint(3, size=nvert )]

        #max_uint32 = (1 << 32) - 1 
        #max_int32 =  (1 << 31) - 1 

        """
        if self.nphotons == 0:return None


        dtype = np.dtype([ 
            ('position_time'   ,          np.float32, 4 ), 
            ('direction_wavelength',      np.float32, 4 ), 
            ('polarization_weight',       np.float32, 4 ), 
            ('ccolor',                    np.float32, 4 ), 
            ('flags',                     np.uint32,  4 ), 
            ('last_hit_triangle',         np.int32,   4 ), 
          ])

        nvert = self.nvert
        log.info( "create_data nphotons %d max_slots %d nvert %d (with slot scaleups) " % (self.nphotons,self.max_slots,nvert) )
        data = np.zeros(nvert, dtype )

        # splaying the start data out into the slots, leaving loadsa free slots
        # NOT A REALISTIC DATA STRUCTURE WITH SO MUCH EMPTY SPACE
        def pack31_( name, a, b ):
            data[name][::self.max_slots,:3] = a
            data[name][::self.max_slots,3] = b
        def pack1_( name, a):
            data[name][::self.max_slots,0] = a
        def pack4_( name, a):
            data[name][::self.max_slots] = a

        pack31_( 'position_time',        self.position ,    self.time )
        pack31_( 'direction_wavelength', self.direction,    self.wavelength )
        pack31_( 'polarization_weight',  self.polarization, self.weight  )
        pack1_(  'flags',                self.flags )
        pack1_(  'last_hit_triangle',    self.last_hit_triangle )
        pack4_(  'ccolor',               self.ccolor) 

        return data


    def _get_ccolor(self):
        """
        #. initialize to red, reset by CUDA kernel
        """
        if self._ccolor is None:
            self._ccolor = np.tile( [1.,0.,0,1.], (self.nphotons,1)).astype(np.float32)    
        return self._ccolor
    ccolor = property(_get_ccolor, doc=_get_ccolor.__doc__)





class DAEPhotonsDataLegacy(DAEPhotonsDataBase):
    numquad = 3
    max_slots = 1 
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
    #print pd
   
    print pd.data



