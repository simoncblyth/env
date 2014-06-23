#!/usr/bin/env python
"""
"""
import logging
log = logging.getLogger(__name__)
import numpy as np

from daephotonsdata import DAEPhotonsDataBase


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



