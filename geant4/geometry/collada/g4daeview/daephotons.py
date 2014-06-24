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

        self._cfg = None
        self._style = event.config.args.style   

        self.numquad = DAEPhotonsData.numquad  # fundamental nature of VBO data structure, not a "parameter"

        self.param = DAEPhotonsParam( event.config)
        self.data = DAEPhotonsData(photons, self.param)
        self.menuctrl = DAEPhotonsMenuController( event.config.rmenu, self.param )
    
        self.renderer = DAEPhotonsRenderer(self, event.scene.chroma ) # pass chroma context to renderer for PyCUDA/OpenGL interop tasks 

        self.propagator = DAEPhotonsPropagator(self, event.scene.chroma, debug=int(event.config.args.debugkernel) ) if self.interop else None
        self.analyzer = DAEPhotonsAnalyzer(self)

        self.propagated = None    
        self._mesh = None
        self._tcut = None
        self.tcut = event.config.args.tcut    

    def deferred_menu_update(self):
        """
        Calling this before GLUT setup, results in duplicates menus 
        """
        log.info("deferred_menu_update")
        self.menuctrl.update_style_menu( self.styles, self.style_callback )

    def style_callback(self, item):
        style = item.title
        log.info("style_callback item %s setting style %s  " % (repr(item),style))
        self.style = style
        self.menuctrl.rootmenu.dispatch('on_needs_redraw')

    def _get_style(self):
        return self._style
    def _set_style(self, style):
        if style == self._style:return
        self._cfg = None        # invalidate dependant 
        self._style = style   
        self.renderer.shaderkey = self.cfg['shaderkey']
        log.info("%s _set_style [%s] %s" % (self.__class__.__name__, style, pprint.pformat(self._cfg)))
    style = property(_get_style, _set_style, doc="Photon presentation style, eg confetti/spagetti/movie/...") 


    def _get_cfg(self):
        if self._cfg is None:
           self._cfg = self.make_cfg()
        return self._cfg
    cfg = property(_get_cfg)



    styles = ['noodles','movie','spagetti','confetti','confetti-0','confetti-1']
    def make_cfg(self):
        """
        :param photonskey: string identifying various techniques to present the photon information

        *slot*
           -1, top slot at max_slots-1
           None, corresponds to using max_slots 1 with slot 0,
           with top slot excluded 
           (ie seeing all steps of the propagation except the artificial 
           interpolated top slot)

        *drawkey*
           `multidraw` is efficient way of in effect doing separate draw calls 
           for each photon (or photon history) eg allowing trajectory line presentation.

           It is so prevalent as without it have no choice but to 
           restrict to slots that will always be present, ie slot 0 and slot -1.
           (unless traversed the entire VBO with selection to skip empties ?)

        Debug tips:

        #. check time dependancy with `udp.py --time 10` etc..

        Animated spagetti, ie LINE_STRIP that animates: not easy 
        as need multidraw dynamic counts with interpolated top slot 
        interposition. Technically challenging but not so informative.
        Would be tractable is could get geometry shader to deal in LINE_STRIPs.

        A point representing ABSORPTIONs would be more useful.


        Live transitions to the "nogeo" shaders `spagetti` 
        and `confetti` are working from all others.  
        The reverse transitions from "nogeo" to "point2line" 
        shaderkey do not work, giving a blank render.

        Presumably an attribute binding problem, not changing a part 
        of opengl state 





        """
        cfg = {}

        style = self.style    
        if style == 'noodles':

           cfg['description'] = "LINE_STRIP direction/polarization at each step of the photon" 
           cfg['drawmode'] = gl.GL_POINTS
           cfg['drawkey'] = "multidraw" 
           cfg['shaderkey'] = "point2line"
           cfg['slot'] = None  

        elif style == 'movie':

           cfg['description'] = "LINE_STRIP direction/polarization that is time interpolated " 
           cfg['drawmode'] = gl.GL_POINTS
           cfg['drawkey'] = "multidraw" 
           cfg['shaderkey'] = "point2line"
           cfg['slot'] = -1  

        elif style == 'confetti':

           cfg['description'] = "POINTS for each step of the photon" 
           cfg['drawmode'] = gl.GL_POINTS
           cfg['drawkey'] = "multidraw" 
           cfg['shaderkey'] = "nogeo"
           cfg['slot'] = None

        elif style == 'confetti-1':

           cfg['description'] = "POINTS for each step of the photon" 
           cfg['drawmode'] = gl.GL_POINTS
           cfg['drawkey'] = "multidraw" 
           cfg['shaderkey'] = "nogeo"
           cfg['slot'] = -1

        elif style == 'confetti-0':

           cfg['description'] = "POINTS for each step of the photon" 
           cfg['drawmode'] = gl.GL_POINTS
           cfg['drawkey'] = "multidraw" 
           cfg['shaderkey'] = "nogeo"
           cfg['slot'] = 0

        elif style == 'spagetti':

           cfg['description'] = "LINE_STRIP trajectory of each photon, " 
           cfg['drawmode'] = gl.GL_LINE_STRIP
           cfg['drawkey'] = "multidraw" 
           cfg['shaderkey'] = "nogeo"
           cfg['slot'] = None

        else:
            assert 0, style


        return cfg 


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
        self.propagated = propagated

        self.menuctrl.update( self.analyzer.history , msg="from propagate" )    

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
        

    def draw(self):
        """
        multidraw mode relies on analyzing the propagated VBO to access the 
        number of propagation steps (actually filled VBO slots, as there will be truncation) 
        """
        if self.photons is None:return

        self.renderer.update_constants()   

        if self.cfg['drawkey'] == 'multidraw':
            self.renderer.multidraw(mode=self.cfg['drawmode'],slot=self.cfg['slot'], 
                                      counts=self.analyzer.counts, 
                                      firsts=self.analyzer.firsts, 
                                   drawcount=self.analyzer.drawcount )
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
        log.debug("_set_photons")
        self.data.photons = photons
        if not photons is None:
            self.renderer.invalidate_buffers()
            self.propagate()
    photons = property(_get_photons, _set_photons) 

    ### other actions #####

    reconfig_handled = ['time','style',]

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

    def _get_time_range(self):
        timerange = self.config.timerange
        if not timerange is None:
            return timerange
        if self.analyzer is None or self.propagated is None:
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


    #def _get_tcut(self):
    #    return self._tcut
    #def _set_tcut(self, tcut):
    #    """
    #    Controlled by up/down trackpad dragging whilst pressing QUOTELEFT at keyboard top left" )
    #    """
    #    self._tcut = np.clip(tcut, 0.00001, 1.) # dont go all the way to zero as cannot then recover
    #
    #        time_range = self.analyzer.time_range 
    #    if time_range is None:
    #        log.debug("cannot act on _set_tcut %s  until event has been propagated and analyzed ", self._tcut )
    #        return  
    #
    #    time = self.time
    #    time += (time_range[1]-time_range[0])*self._tcut + time_range[0] 
    #
    #    log.info("_set_tcut %s %s => %s " % (self._tcut, repr(time_range), time ))
    #    self.time = time
    #tcut = property(_get_tcut, _set_tcut, doc=_set_tcut.__doc__ )



    def __repr__(self):
        return "%s " % (self.__class__.__name__)




if __name__ == '__main__':
    pass




