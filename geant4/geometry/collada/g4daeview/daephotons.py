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
from daephotonsanalyzer import DAEPhotonsAnalyzer, DAEPhotonsPropagated
from daephotonsstyler import DAEPhotonsStyler
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
        :param event: `DAEEventBase` instance
        """ 
        self.event = event       
        self.interop = not event.scene.chroma.dummy
        self.config = event.config      

        geometry = event.scene.geometry
        for att in "chroma_material_map chroma_surface_map chroma_process_map".split():
            if hasattr(geometry,att):
                setattr(self,att,getattr(geometry,att))
            pass
        pass

        self._style = event.config.args.style   

        self.numquad = DAEPhotonsData.numquad  # fundamental nature of VBO data structure, not a "parameter"

        self.styler = DAEPhotonsStyler()

        self.param = DAEPhotonsParam( event.config)
        self.data = DAEPhotonsData(photons, self.param)
        self.menuctrl = DAEPhotonsMenuController( event.config.rmenu, self.param )
    
        self.renderer = DAEPhotonsRenderer(self, event.scene.chroma ) # pass chroma context to renderer for PyCUDA/OpenGL interop tasks 
        presenter = self.renderer.presenter
        if not presenter is None:
            presenter.time   = event.config.args.time
            presenter.cohort = event.config.args.cohort
        pass
        self.material = event.config.args.material    
        self.surface  = event.config.args.surface
        self.mode = event.config.args.mode    

        self.propagator = DAEPhotonsPropagator(self, event.scene.chroma, debug=int(event.config.args.debugkernel) ) if self.interop else None
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

    def deferred_menu_update(self):
        """
        Calling this before GLUT setup, results in duplicated menus 
        """
        if not self.interop:return
        self.menuctrl.update_style_menu( self.styler.style_names_menu, self.style_callback )
        self.menuctrl.update_material_menu( self.material_pairs(), self.material_callback )


    def material_pairs(self):
         return self.analyzer.get_material_pairs(self.chroma_material_map)

    def material_callback(self, item):
        matname = item.title
        matcode = item.extra['matcode']
        log.info("material_callback matname %s matindex %s  " % (matname, matcode) )
        self.material = matcode
        self.menuctrl.rootmenu.dispatch('on_needs_redraw')

    def style_callback(self, item):
        style = item.title
        self.style = style
        self.menuctrl.rootmenu.dispatch('on_needs_redraw')

    def special_callback(self, item):
        sid = int(item.extra['sid'])
        self.param.sid = sid
        self.menuctrl.rootmenu.dispatch('on_needs_redraw')

    def _get_style(self):
        return self._style
    def _set_style(self, style):
        if style == self._style:return
        self._style = style   
    style = property(_get_style, _set_style, doc="Photon presentation style, eg confetti/spagetti/movie/...") 

    #def _get_cfg(self):
    #    return self.styler.get(self.style)
    #cfg = property(_get_cfg)

    def _get_cfglist(self):
        return self.styler.get_list(self.style)
    cfglist = property(_get_cfglist)



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
        if self.photons is None:return

        #max_slots = self.config.args.max_slots
        max_slots = self.data.max_slots

        vbo = self.renderer.pbuffer   

        self.propagator.update_constants()   

        if not self.config.args.propagate:
            log.warn("propagation is inhibited by config: -P/--nopropagate ")  
        else:
            log.warn("propagation proceeding")  
            self.propagator.interop_propagate( vbo, max_slots=max_slots ) 
        pass 

        propagated = vbo.read()

        self.analyzer( propagated )

        if self.config.args.debugpropagate:
            self.analyzer.write_propagated(self.propagator.ctx.seed, self.event.loaded, wipepropagate=self.config.args.wipepropagate)
        pass
        self.menuctrl.update_propagated( self.analyzer , special_callback=self.special_callback, msg="from propagate" )    

        nphotons = self.data.nphotons
        last_slot = -2
        last_slot_indices = np.arange(nphotons)*max_slots + (max_slots+last_slot)

        #p = propagated[::max_slots]  ## slot 0 
        p = propagated[last_slot_indices]

        r = np.zeros( (len(p),4,4), dtype=np.float32 )  

        r[:,0,:4] = p['position_time'] 
        r[:,1,:4] = p['direction_wavelength'] 
        r[:,2,:4] = p['polarization_weight'] 
        r[:,3,:4] = p['last_hit_triangle'].view(r.dtype) # must view as target type to avoid coercion of int32 data into float32

        #photon_id  = r[:,3,0]     #    
        #spare      = r[:,3,1]     #    
        #flags      = r[:,3,2]     # history  
        channel_id = r[:,3,3]     # pmtid 

        return r[channel_id > 0]   


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
            self.lastpropagated = None
            self.renderer.invalidate_buffers()
            lastpropagated = self.propagate()
            self.lastpropagated = lastpropagated
        pass


    photons = property(_get_photons, _set_photons, doc="NB the act of setting photons performs the propagation" ) 

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


    ## step material selection

    def _set_material(self, names):
        presenter = self.renderer.presenter
        if presenter is None:
            log.warn("cannot set material selection constants when renderer.presenter is not enabled")
            return
        pass
        codes = self.chroma_material_map.convert_names2codes(names)
        log.info("_set_mate %s => %s " % (names, codes))
        presenter.material = codes
    def _get_material(self):
        presenter = self.renderer.presenter
        if presenter is None:
            return None
        pass
        codes = presenter.material
        names = self.chroma_material_map.convert_codes2names(codes)
        log.info("_get_mate %s => %s " % (codes, names))
        return names
    material = property(_get_material, _set_material, doc="setter copies material selection integers into GPU quad g_mate  getter returns cached value " )



    def _set_surface(self, names):
        presenter = self.renderer.presenter
        if presenter is None:
            log.warn("cannot set material selection constants when renderer.presenter is not enabled")
            return
        pass
        codes = self.chroma_surface_map.convert_names2codes(names)
        log.info("_set_surface %s => %s " % (names, codes))
        presenter.surface = codes
    def _get_surface(self):
        presenter = self.renderer.presenter
        if presenter is None:
            return None
        pass
        codes = presenter.surface
        names = self.chroma_surface_map.convert_codes2names(codes)
        log.info("_get_surface %s => %s " % (codes, names))
        return names
    surface = property(_get_surface, _set_surface, doc="surface: setter copies selection integers into GPU quad g_surf  getter returns cached value " )







    def _set_mode(self, mode):
        presenter = self.renderer.presenter
        if presenter is None:
            log.warn("cannot set mode selection constants when renderer.presenter is not enabled")
            return
        pass
        presenter.mode = mode
    def _get_mode(self):
        presenter = self.renderer.presenter
        if presenter is None:
            return None
        pass
        return presenter.mode
    mode = property(_get_mode, _set_mode, doc="mode: setter copies mode control integers into GPU quad g_mode  getter returns cached value " )






    def __repr__(self):
        return "%s " % (self.__class__.__name__)




if __name__ == '__main__':
    pass




