#!/usr/bin/env python
"""

"""
import logging
log = logging.getLogger(__name__)

from daegeometry import DAEMesh 
from daephotonsparam import DAEPhotonsParam
from daephotonsmenuctrl import DAEPhotonsMenuController
from daephotonsrenderer import DAEPhotonsRenderer
from daephotonsdata import DAEPhotonsData, DAEPhotonsDataLegacy


class DAEPhotons(object):
    """
    Coordinator class handling photon presentation, 
    specifics belong in constituents

    #. `data` photon data manipulations/selections controlled by
        
       * `data.photons`
       * `data.param`

    #. `menuctrl` GLUT menus  
    #. `renderer` OpenGL drawing

    The photons and param constituens use a layered 
    property pattern where setters at this level deal with 
    things like menu updates, and the DAEPhotonData deal with
    more fundamental things.

    """ 
    def __init__(self, photons, event ):
        """
        :param photons: ChromaPhotonList instance
        """ 
        self.event = event       

        param = DAEPhotonsParam( event.config)
        datacls = DAEPhotonsDataLegacy if event.config.args.legacy else DAEPhotonsData
        self.numquad = datacls.numquad  # not really a parameter, rather a fundamental feature of data structure in use
        self.data = datacls(photons, param)

        self.menuctrl = DAEPhotonsMenuController( event.config.rmenu, self.param )
        self.renderer = DAEPhotonsRenderer(self, event.scene.chroma) # pass chroma context to renderer for PyCUDA/OpenGL interop tasks 
        self._mesh = None

    def _get_qcount(self):
        """
        Where to put this ?
        """
        return int(self.data.nphotons*self.event.qcut)
    qcount = property(_get_qcount, doc="Photon count modulated by qcut which varies between 0 and 1. Used for partial drawing based on a sorted quantity." ) 

    vertices     = property(lambda self:self.data.position)  # allows to be treated like DAEMesh 

    def _get_mesh(self):
        if self._mesh is None:
            self._mesh = DAEMesh(self.data.position)
        return self._mesh
    mesh = property(_get_mesh)

    def draw(self):
        if self.photons is None:return
        self.renderer.draw()

    def _get_photons(self):
        return self.data.photons 
    def _set_photons(self, photons):
        self.data.photons = photons
        if not photons is None:
            self.renderer.invalidate_buffers()
            self.menuctrl.update( photons )    
    photons = property(_get_photons, _set_photons) 

    def _get_param(self):
        return self.data.param 
    def _set_param(self, param):
        self.data.param = param
        if not param is None:
            self.menuctrl.update_history_menu( self.photons )   
    param = property(_get_param, _set_param) 


    def __repr__(self):
        return "%s " % (self.__class__.__name__)

    def reconfig(self, conf):
        """
        This is called to handle external messages such as::

            udp.py --fpholine 100

        This formerly forced buffer invalidation, but following move
        to shader rendering, can just update uniforms.
        """
        log.info("reconfig %s " % repr(conf))
        update = self.param.reconfig(conf)
        #
        #if update:
        #    self.renderer.invalidate_buffers()
        #




if __name__ == '__main__':
    pass




