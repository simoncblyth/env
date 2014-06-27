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
from daephotonsdata import DAEPhotonsData
from daephotonspropagator import DAEPhotonsPropagator
from daephotonsanalyzer import DAEPhotonsAnalyzer
from daephotonsstyler import DAEPhotonsStyler


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
        self.interop = not event.scene.chroma.dummy
        self.config = event.config      

        self._style = event.config.args.style   

        self.numquad = DAEPhotonsData.numquad  # fundamental nature of VBO data structure, not a "parameter"

        self.styler = DAEPhotonsStyler(self)

        self.param = DAEPhotonsParam( event.config)
        self.data = DAEPhotonsData(photons, self.param)
        self.menuctrl = DAEPhotonsMenuController( event.config.rmenu, self.param )
    
        self.renderer = DAEPhotonsRenderer(self, event.scene.chroma ) # pass chroma context to renderer for PyCUDA/OpenGL interop tasks 
        self.renderer.presenter.time   = event.config.args.time
        self.renderer.presenter.cohort = event.config.args.cohort


        self.propagator = DAEPhotonsPropagator(self, event.scene.chroma, debug=int(event.config.args.debugkernel) ) if self.interop else None
        self.analyzer = DAEPhotonsAnalyzer(self.config.args.max_slots)

        self._mesh = None
        self._tcut = None
        self.tcut = event.config.args.tcut    



    def deferred_menu_update(self):
        """
        Calling this before GLUT setup, results in duplicated menus 
        """
        self.menuctrl.update_style_menu( self.styler.style_names, self.style_callback )

    def style_callback(self, item):
        style = item.title
        self.style = style
        self.menuctrl.rootmenu.dispatch('on_needs_redraw')

    def _get_style(self):
        return self._style
    def _set_style(self, style):
        if style == self._style:return
        self._style = style   
        #self.renderer.shaderkey = self.cfg['shaderkey']
    style = property(_get_style, _set_style, doc="Photon presentation style, eg confetti/spagetti/movie/...") 

    def _get_cfg(self):
        return self.styler.get(self.style)
    cfg = property(_get_cfg)

    def _get_cfglist(self):
        return self.styler.get_list(self.style)
    cfglist = property(_get_cfglist)



    ### primary actions #####

    def propagate(self, max_steps=100):
        """
        #. performs photon propagation using the chroma fork propagate_vbo.cu:propagate_vbo 
        #. reads back the VBO from OpenGL into numpy array
        #. analyze characteristics of the propagation

        Option `--debugpropagate` persists the numpy propagated array into 
        file `propagated.npz`. Access the array as shown below, or see standalone `propagated.py`::

           with np.load('propagated.npz') as npz:
               a = npz['propagated']

        """
        if self.photons is None:return

        vbo = self.renderer.pbuffer   

        self.propagator.update_constants()   
        self.propagator.interop_propagate( vbo, max_steps=max_steps )
            
        propagated = vbo.read()
        self.analyzer( propagated )
        if self.config.args.debugpropagate:
            self.analyzer.write_propagated()
        pass
        self.menuctrl.update( self.analyzer.history , msg="from propagate" )    

       

    def draw(self):
        """
        multidraw mode relies on analyzing the propagated VBO to access the 
        number of propagation steps (actually filled VBO slots, as there will be truncation) 
        """
        if self.photons is None:return
        self.renderer.update_constants()   
        for cfg in self.cfglist:
            self.drawcfg( cfg )


    def drawcfg(self, cfg ): 
        self.renderer.shaderkey = cfg['shaderkey']
        if cfg['drawkey'] == 'multidraw':
            counts, firsts, drawcount = self.analyzer.counts_firsts_drawcount 
            self.renderer.multidraw(mode=cfg['drawmode'],slot=cfg['slot'], 
                                      counts=counts, 
                                      firsts=firsts, 
                                   drawcount=drawcount, extrakey=cfg['extrakey'] )
        else:
            self.renderer.draw(mode=cfg['drawmode'],slot=cfg['slot'])


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
        log.debug("_set_photons")
        self.data.photons = photons
        if not photons is None:
            self.renderer.invalidate_buffers()
            self.propagate()
    photons = property(_get_photons, _set_photons) 

    ### other actions #####

    reconfig_properties = ['time','style','timerange','cohort',]

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


    ### time control ######## 

    def _get_time_range(self):
        timerange = self.config.timerange
        if not timerange is None:
            return timerange
        if self.analyzer is None or self.analyzer.propagated is None:
            return None
        return self.analyzer.time_range
    time_range = property(_get_time_range)

    def _set_time(self, time):
        presenter = self.renderer.presenter
        if presenter is None:
            log.warn("cannot set time when renderer.presenter is not enabled")
            return
        time_range = self.time_range
        if time_range is None:
           time_range = [0,1000,]    
           log.warn("using default time_range %s ns " % repr(time_range))
        pass
        presenter.time = np.clip(time, time_range[0], time_range[1] )

    def _get_time(self):
        presenter = self.renderer.presenter
        return 0. if presenter is None else presenter.time
    time = property(_get_time, _set_time, doc="setter copies time into GPU constant g_anim.x, getter returns cached value " )


    def _set_cohort(self, cohort):
        presenter = self.renderer.presenter
        if presenter is None:
            log.warn("cannot set cohort when renderer.presenter is not enabled")
            return
        presenter.cohort = cohort
    def _get_cohort(self):
        presenter = self.renderer.presenter
        return None if presenter is None else presenter.cohort
    cohort = property(_get_cohort, _set_cohort, doc="setter copies cohort begin/end times into GPU constants g_anim.y,z , getter returns cached value " )



    def _set_time_fraction(self, time_fraction):
        time_range = self.time_range
        if time_range is None:
            return 
        time = time_range[0] + time_fraction*(time_range[1] - time_range[0])
        self.time = time 
    def _get_time_fraction(self):
        time_range = self.time_range
        if time_range is None:
            return None
        return ( self.time - time_range[0] ) / (time_range[1] - time_range[0] )
    time_fraction = property(_get_time_fraction, _set_time_fraction )




    def time_to(self, x, y, dx, dy):
        """
        Use for real propagation time control, not the fake time of initial photon 
        variety.
        """
        tfactor = 10. 
        tdelta = tfactor*dy
        self.time += tdelta
        #log.info("time_to x %s y %s dx %s dy %s ===> tfactor %s tdelta  %s ===> time %s " % (x,y,dx,dy, tfactor, tdelta, self.time))



    def __repr__(self):
        return "%s " % (self.__class__.__name__)




if __name__ == '__main__':
    pass




