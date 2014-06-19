#!/usr/bin/env python
"""

"""
import logging
log = logging.getLogger(__name__)

import numpy as np
from daegeometry import DAEMesh 
from daephotonsparam import DAEPhotonsParam
from daephotonsmenuctrl import DAEPhotonsMenuController
from daephotonsrenderer import DAEPhotonsRenderer
from daephotonsdata import DAEPhotonsData, DAEPhotonsDataLegacy
from daephotonspropagator import DAEPhotonsPropagator


class DAEPhotons(object):
    """
    Coordinator class handling photon presentation, 
    specifics belong in constituents

    Constituents
    ~~~~~~~~~~~~~

    #. data, DAEPhotonsData 
    #. menuctrl, DAEPhotonsMenuController: GLUT menus  
    #. renderer, DAEPhotonsRenderer: OpenGL drawing

    Read/Write properties:
    ~~~~~~~~~~~~~~~~~~~~~~

    #. photons
    #. param 

    Using a layered property pattern where setters at 
    this level deal with things like menu updates, 
    and the lower level `data` DAEPhotonData setters 
    deal with more fundamental things like buffer invalidation.

    Readonly properties:
    ~~~~~~~~~~~~~~~~~~~~~

    #. vertices
    #. qcount
    #. mesh

    actions
    ~~~~~~~~

    #. draw, passed to renderer
    #. reconfig, passed to param


    """ 
    def __init__(self, photons, event ):
        """
        :param photons: `chroma.event.Photons` instance (or fallback)
        :param event: `DAEEvent` instance
        """ 
        self.event = event       
        self.config = event.config      

        param = DAEPhotonsParam( event.config)
        datacls = DAEPhotonsDataLegacy if event.config.args.legacy else DAEPhotonsData
        self.numquad = datacls.numquad  # not really a parameter, rather a fundamental feature of data structure in use

        self.interop = not event.scene.chroma.dummy
        self.data = datacls(photons, param)
        self.menuctrl = DAEPhotonsMenuController( event.config.rmenu, self.param )
        self.renderer = DAEPhotonsRenderer(self, event.scene.chroma) # pass chroma context to renderer for PyCUDA/OpenGL interop tasks 
        self.propagator = DAEPhotonsPropagator(self, event.scene.chroma, debug=int(event.config.args.debugkernel) ) if self.interop else None
        self.propagated = None    

        self._mesh = None


    ### primary actions #####

    def propagate(self, max_steps=100):
        """
        When option `--debugpropagate` is used the propagated VBO
        is read back into a numpy array and persisted to file `propagated.npz`.

        Access::

           with np.load('propagated.npz') as npz:
               a = npz['propagated']

        """
        if self.photons is None:return

        vbo = self.renderer.pbuffer   
        self.propagator.update_constants()   
        self.propagator.interop_propagate( vbo, max_steps=max_steps )

        if self.config.args.debugpropagate:
            propagated = vbo.read()
            path = "propagated.npz"
            log.info("propagate completed, save VBO readback into %s " % path )
            print propagated
            print propagated.dtype
            print propagated.size
            print propagated.itemsize
            np.savez_compressed(path, propagated=propagated)
        
        self.propagated = propagated


    def draw(self, slot=-1):
        """
        :param slot: -1 means the reserved slot at max_slots-1
        """
        if self.photons is None:return
        self.renderer.draw(slot=slot)


    #### readonly properties #####

    vertices     = property(lambda self:self.data.position)  # allows to be treated like DAEMesh  

    def _get_qcount(self):
        """
        Photon count modulated by qcut which varies between 0 and 1. 
        Used for partial drawing based on a sorted quantity.
        """ 
        return int(self.data.nphotons*self.event.qcut)
    qcount = property(_get_qcount, doc=_get_qcount.__doc__) 

    def _get_mesh(self):
        if self._mesh is None:
            self._mesh = DAEMesh(self.data.position)
        return self._mesh
    mesh = property(_get_mesh)

    #### read/write  properties #####

    def _get_photons(self):
        return self.data.photons 
    def _set_photons(self, photons):
        self.data.photons = photons
        if not photons is None:
            self.renderer.invalidate_buffers()
            self.propagate()
            self.menuctrl.update( photons )    
    photons = property(_get_photons, _set_photons) 

    def _get_param(self):
        return self.data.param 
    def _set_param(self, param):
        self.data.param = param
        if not param is None:
            self.menuctrl.update_history_menu( self.photons )   
    param = property(_get_param, _set_param) 


    ### other actions #####

    def reconfig(self, conf):
        """
        This is called to handle external messages such as::

            udp.py --fpholine 100

        Parameter reconfig updates formerly forced `renderer.invalidate_buffers()`
        but following migration to shader rendering, 
        can just update uniforms.
        """
        log.info("reconfig %s " % repr(conf))
        update = self.param.reconfig(conf)

    def __repr__(self):
        return "%s " % (self.__class__.__name__)




if __name__ == '__main__':
    pass




