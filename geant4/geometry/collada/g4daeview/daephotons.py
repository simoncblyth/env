#!/usr/bin/env python
"""

"""
import logging, pprint
log = logging.getLogger(__name__)

import OpenGL.GL as gl

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

        cfg = self.configure(event.config.args.photons)
        log.info("%s %s" % (self.__class__.__name__, pprint.pformat(cfg)))
        self.cfg = cfg

        param = DAEPhotonsParam( event.config)
        datacls = DAEPhotonsDataLegacy if event.config.args.legacy else DAEPhotonsData
        self.numquad = datacls.numquad  # not really a parameter, rather a fundamental feature of data structure in use

        self.interop = not event.scene.chroma.dummy
        self.data = datacls(photons, param)
        self.menuctrl = DAEPhotonsMenuController( event.config.rmenu, self.param )
        self.renderer = DAEPhotonsRenderer(self, event.scene.chroma, cfg ) # pass chroma context to renderer for PyCUDA/OpenGL interop tasks 
        self.propagator = DAEPhotonsPropagator(self, event.scene.chroma, debug=int(event.config.args.debugkernel) ) if self.interop else None
        self.propagated = None    

        self._mesh = None


    def configure(self, photonskey ):
        """
        :param photonskey: string identifying various techniques to present the photon information

        Meaning of slot values

        * slot -1, the reserved slot at max_slots-1
        * slot None, using max_slots 1 with slot 0, ie seeing all steps of the propagation at once

        """
        cfg = {}

        if photonskey == 'noodlesoup':

           cfg['description'] = "Generated direction/polarization LINE_STRIP at each step of the photon" 
           cfg['drawmode'] = gl.GL_POINTS
           cfg['drawkey'] = "multidraw" 
           cfg['shaderkey'] = "point2line"
           cfg['slot'] = None

        elif photonskey == 'movie':

           cfg['description'] = "" 
           cfg['drawmode'] = gl.GL_POINTS
           cfg['drawkey'] = "multidraw" 
           cfg['shaderkey'] = "point2line"
           cfg['slot'] = -1

        elif photonskey == 'confetti':

           cfg['description'] = "POINTS for each step of the photon" 
           cfg['drawmode'] = gl.GL_POINTS
           cfg['drawkey'] = "multidraw" 
           cfg['shaderkey'] = "nogeo"
           cfg['slot'] = None

        elif photonskey == 'spagetti':

           cfg['description'] = "LINE_STRIP trajectory of each photon" 
           cfg['drawmode'] = gl.GL_LINE_STRIP
           cfg['drawkey'] = "multidraw" 
           cfg['shaderkey'] = "nogeo"
           cfg['slot'] = None

        else:
            assert 0, photonskey


        return cfg 


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
            
        propagated = vbo.read()
        self.analyse_propagation( propagated )

        self.propagated = propagated

        if self.config.args.debugpropagate:
            path = "propagated.npz"
            log.info("propagate completed, save VBO readback into %s " % path )
            print propagated
            print propagated.dtype
            print propagated.size
            print propagated.itemsize
            np.savez_compressed(path, propagated=propagated)
        else:
            pass
        

    def analyse_propagation(self, propagated):
        """
        Interpret counts layed down at tail of propagate_vbo.cu:propagate_vbo

        What an "item" is for glDrawArrays depends on the strides setup in DAEVertexAttrib

        Attempts to draw a single (or few) photon histories are failing, ie
        still see loadsa lines and for everything other than `--debugphoton 0` get
        an Abort Trap::

            g4daeview.sh --with-chroma --load 1 --debugshader --max-slots 10 --debugkernel --debugphoton 0 --debugpropagate 
            g4daeview.sh --with-chroma --load 1 --debugshader --max-slots 10 --debugkernel --debugphoton 1 --debugpropagate 

        So trying alternate means to make vertices disappear by scooting them off to infinity.

        #. NB this is based on a very wasteful and truncating array structure


        """
        log.info("analyse_propagation")
        a = propagated
        if a is None:return
        max_slots = self.data.max_slots

        field = 'last_hit_triangle'
        lht = a[field][::max_slots,0]
        photon_id = a[field][::max_slots,1]
        steps = a[field][::max_slots,2]
        slots = a[field][::max_slots,3]

        #assert np.all( lht == -1 )  no longer the case, as are now putting last slot result into slot 0
        assert np.all(np.arange(0,len(photon_id),dtype=np.int32) == photon_id)
        assert np.all( steps == slots )

        counts = np.clip( slots, 0, max_slots-2 ) + 1               # counts of numquad photon records 
        firsts  = np.arange(len(photon_id), dtype='i')*max_slots   # multipled by numquad ?
        assert len(counts) == len(firsts) == len(photon_id)

        n = 0 # > 0 causes Abort Trap
        if self.config.args.debugkernel and n > 0:
            index = np.where( photon_id == self.config.args.debugphoton )[0][0] 
            self.counts = counts[index:index+n]
            self.firsts = firsts[index:index+n]
            self.drawcount = n
        else:
            self.counts = counts
            self.firsts = firsts
            self.drawcount = len(photon_id)
        pass
        log.info( " counts %s " % str(self.counts))
        log.info( " firsts %s " % str(self.firsts))
        log.info( " drawcount %s " % str(self.drawcount))


    def draw(self):
        """
        """
        if self.photons is None:return

        if self.cfg['drawkey'] == 'multidraw':
            self.renderer.multidraw(mode=self.cfg['drawmode'],slot=self.cfg['slot'], counts=self.counts, firsts=self.firsts, drawcount=self.drawcount )
        else:
            self.renderer.draw(mode=self.cfg['drawmode'],slot=self.cfg['slot'])


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

    reconfig_handled = ['time',]

    def reconfig(self, conf):
        """
        This is called to handle external messages such as::

            udp.py --fpholine 100

        Parameter reconfig updates formerly forced `renderer.invalidate_buffers()`
        but following migration to shader rendering, 
        can just update uniforms.
        """
        update = False
        unhandled = []
        for k, v in conf:
            if k in self.reconfig_handled:
                setattr(self, k, v ) 
                update = True
            else:
                unhandled.append((k,v,))
            pass 

        param_update = self.param.reconfig(unhandled)
        return update or param_update


    def _set_time(self, time):
        presenter = self.renderer.presenter
        if presenter is None:
            log.warn("cannot set time when renderer.presenter is not enabled")
        else:
            presenter.time = time
    def _get_time(self):
        presenter = self.renderer.presenter
        return None if presenter is None else presenter.time
    time = property(_get_time, _set_time, doc="setter copies time into GPU constant g_anim.x, getter returns cached value " )

    def __repr__(self):
        return "%s " % (self.__class__.__name__)




if __name__ == '__main__':
    pass




