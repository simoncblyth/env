#!/usr/bin/env python
"""

"""
import logging, pprint
log = logging.getLogger(__name__)

import OpenGL.GL as gl

import numpy as np

from daephotonsanalyzer import DAEPhotonsAnalyzer, DAEPhotonsPropagated
from daephotonspropagator import DAEPhotonsPropagator

from daedrawable import DAEDrawable
from env.g4dae.types import VBOPhoton

class NPY(np.ndarray):pass


class DAEPhotons(DAEDrawable):
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
    def __init__(self, _array, event ):
        """
        :param photons: typically None, set later
        :param event: `DAEEventBase` instance
        """ 
        DAEDrawable.__init__(self, _array, event ) 

        self.numquad = VBOPhoton.numquad  # fundamental nature of VBO data structure, not a "parameter"
        assert self.config.args.numquad == self.numquad

        self.propagator = DAEPhotonsPropagator(self, event.scene.chroma, debug=int(event.config.args.debugkernel) ) if self.interop else None

        # hmm max_slots can now be changed per batch  
        self.analyzer = DAEPhotonsAnalyzer(self.config.args.max_slots)
        self.tpropagated = DAEPhotonsPropagated( None, self.config.args.max_slots )

        # lastpropagated photons object used for communication with the responder
        # _set_photons clears this  
        self.lastpropagated = None   

        self._mesh = None
        self._tcut = None
        self.tcut = event.config.args.tcut    


    def clicked_point(self, click, live=True):
        """
        :param click: world coordinate xyz of point clicked
        :param live: when True pull the VBO content off the GPU in order to 
                     access the t-interpolated slot -1 positions (requires --pullvbo config enabled)
                     when False use last photon positions 
        """
        if not self.interop:return
        if self.config.args.pullvbo and live:
            vbo = self.renderer.pbuffer   
            if vbo is None:return 
            if not self.tpropagated.is_enabled:return 

            self.tpropagated( vbo.read() ) 
            index = self.tpropagated.t_nearest_photon( click ) 
            self.tpropagated.summary(index, material_map=self.chroma_material_map, process_map=self.chroma_process_map)
        else:
            if not self.analyzer.is_enabled:return 
            index = self.analyzer.nearest_photon(click)
            self.analyzer.summary(index, material_map=self.chroma_material_map, process_map=self.chroma_process_map)
        pass

        log.info("clicked_point %s => index %s " % (repr(click),index))
        self.param.pid = index



    ### primary actions #####

    def propagate(self, max_steps=100):
        """
        :param max_steps:
        :return propagated: photon data instance

        TODO: 

        Translate propagated photons into the 
        ndarray shape and structure expected by:

        #. invoking caller all the way back in Geant4/C++ 
        #. transport machinery inbetween here and there 


        propagate actions
        ~~~~~~~~~~~~~~~~~~

        #. performs photon propagation using the chroma fork propagate_vbo.cu:propagate_vbo 
        #. reads back the VBO from OpenGL into numpy array
        #. analyze characteristics of the propagation

        Option `--debugpropagate` persists the numpy propagated array into 
        into path next to the originating event file with name including 
        the seed eg `propagated-0.npz`. 
        Access the array using eg `daephotonsanalyser.sh --load 1`


        **CAUTION** there is different propagation invoked in daedirectpropagator
        """
        log.info("propagate")
        if self.array is None:return

        max_slots = self.param.max_slots

        vbo = self.renderer.pbuffer   



        self.propagator.update_constants()   

        if not self.config.args.propagate:
            log.warn("propagation is inhibited by config: -P/--nopropagate ")  
        else:
            log.warn("propagation proceeding max_slots %s " % max_slots )  
            self.propagator.interop_propagate( vbo, max_slots=max_slots ) 
        pass 

        propagated = vbo.read()

        if self.config.args.analyze:
            self.analyzer( propagated, checks=True )
            self.analyzer.check_history()

            if self.config.args.debugpropagate:
                self.analyzer.write_propagated(self.propagator.ctx.seed, self.event.loaded, wipepropagate=self.config.args.wipepropagate)
            pass
            self.menuctrl.update_propagated( self.analyzer , special_callback=self.special_callback, msg="from propagate" )    
        else:
            log.warn("not analyzing propagated photons")
        pass

        nitem = len(self.array)
        last_slot = -2
        last_slot_indices = np.arange(nitem)*max_slots + (max_slots+last_slot)

        p = propagated[last_slot_indices]   # VBO branch

        r = VBOPhoton.from_vbo_propagated(p)
        hits = r.hits 

        ## respond with all propagated photons or only those that register a hit on PMT
        if self.propagator.ctx.parameters['hit']:
            response = hits
        else:
            response = r
        pass
        assert self.event.scene.chroma == self.propagator.ctx 

        metadata = {}
        metadata['test'] = { 
                               'npropagated':len(r), 
                                 'nhits':len(hits), 
                                 'nresponse':len(response), 
                                'propagator':"daephotons" 
                          }
        metadata['geometry'] = self.event.scene.chroma.metadata
        response.meta = [metadata]
        return response



    #### read/write  properties #####

    def handle_array(self, _array):
        log.debug("handle_array")
        vbop = VBOPhoton.vbo_from_array(_array, self.param.max_slots)
        return vbop

    def post_handle_array(self):
        log.info("post_handle_array")
        self.lastpropagated = None
        self.renderer.invalidate_buffers()
        lastpropagated = self.propagate()
        self.lastpropagated = lastpropagated


    ### other actions #####

    reconfig_properties = ['time','style','timerange','cohort','material','mode','surface',]

    def reconfig(self, conf):
        """
        This is called to handle external messages such as::

            udp.py --fpholine 100

        Parameter reconfig updates formerly forced `renderer.invalidate_buffers()`
        but following migration to shader rendering, 
        can just update uniforms.
        """
        log.info("reconfig %s " % repr(conf))

        update = False
        unhandled = []
        for k, v in conf:
            if k == 'timerange':
                self.config.timerange = v   # because fvec done there, hmm should move such stuff to point of use ?
            elif k in self.reconfig_properties:
                setattr(self, k, v ) 
                update = True
            else:
                unhandled.append((k,v,))
            pass 

        param_update = self.param.reconfig(unhandled)
        return update or param_update


    ## step material selection


    def __repr__(self):
        return "%s " % (self.__class__.__name__)




if __name__ == '__main__':
    pass




