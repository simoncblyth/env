#!/usr/bin/env python

import logging, datetime
log = logging.getLogger(__name__)
import numpy as np

from daephotons import DAEPhotons
from daegenstep import DAEGenstep

from daeeventlist import DAEEventList , DAEEventListMenu
from daemenu import DAEMenu
from daeanimator import DAEAnimator

from env.g4dae.types import Photon, G4Step, NPY


class DAEEventMenu(DAEMenu):
    def __init__(self, config, handler):
        DAEMenu.__init__(self, "event")
        self.add("reload",handler.reload_)
        self.add("loadnext",handler.loadnext)
        self.add("loadprev",handler.loadprev)


class DAEEvent(object):
    """
    """
    def __init__(self, config, scene ):
        self.config = config
        self.scene = scene
        self.loaded = None
        pass
        self.qcut = config.args.qcut 
        self.bbox_cache = None

        log.info("********* scene.event.dphotons ")

         
        
        self.dphotons = DAEPhotons( None, self )
        self.dgenstep = DAEGenstep( None, self )

        self.menuholder = self.dphotons

        log.info("********* scene.event.dphotons DONE ")
        self.objects = []
        self.eventlist = DAEEventList(config.path_template)

        # dont like this menu setup here, move into constituent controllers

        rmenu = self.config.rmenu 

        event_menu = DAEEventMenu( config, self )
        rmenu.addSubMenu(event_menu) 

        eventlist_menu = DAEEventListMenu(self.eventlist, self.eventlist_callback )
        rmenu.addSubMenu(eventlist_menu) 

        self.set_toggles()
        self.animator = DAEAnimator(config.args.timeperiod)

    def apply_launch_config(self):
        """
        #. now invoked externally and deferred to last moment after GLUT setup 
        """
        log.info("apply_launch_config")
        launch_config = [] 
        if not self.config.args.load is None:
            launch_config.append( ['load',self.config.args.load])
        if not self.config.args.key is None:
            launch_config.append( ['key',self.config.args.key])
        pass
        if len(launch_config) > 0:
            self.reconfig(launch_config)
        pass
        self.menuholder.deferred_menu_update()

    def eventlist_callback(self, item):
        path = item.extra['path'] 
        self.load(path)

    def set_toggles(self):
        self.animate = False

    def toggle_animate(self):
        self.toggle('animate')

    def clicked_point(self, click):
        """
        :param click: world coordinate xyz of point clicked
        """
        if self.config.args.click:
            self.dphotons.clicked_point( click )

    def toggle(self, name):
        log.info("toggle %s" % name )
        setattr( self, name , not getattr(self, name)) 

    def animation_period(self, factor ):   
        self.animator.change_period(factor)

    def tick(self, dt):
        time_fraction, bump = self.animator() 
        #log.info("event tick  %s" % time_fraction ) 
        self.dphotons.time_fraction = time_fraction 

    def _get_time(self):
        return 0
        #return self.dphotons.time
    time = property(_get_time, doc="Animation time")


    def __repr__(self):
        return "t %5.2f" % self.time
    def _get_qcut(self):
        return self._qcut
    def _set_qcut(self, qcut):
        self._qcut = np.clip(qcut, 0.00001, 1.) # dont go all the way to zero as cannot then recover
    qcut = property(_get_qcut, _set_qcut)

    def make_bbox_cache(self):
        """
        Hmm problem with a photons bbox is that its often too big to be useful
        """
        bbox_cache = np.empty((len(self.objects),6))    
        for i, obj in enumerate(self.objects):
            bbox_cache[i] = obj.lower_upper
        pass
        self.bbox_cache = bbox_cache

    def find_bbox_object(self, xyz):
        """
        :param xyz: world frame coordinate

        Find indices of all objects that contain the world frame coordinate provided  
        """
        if self.bbox_cache is None:
            self.make_bbox_cache() 
        x,y,z = xyz 
        b = self.bbox_cache
        f = np.where(
              np.logical_and(
                np.logical_and( 
                  np.logical_and(x > b[:,0], x < b[:,3]),
                  np.logical_and(y > b[:,1], y < b[:,4]) 
                              ),  
                  np.logical_and(z > b[:,2], z < b[:,5])
                            )   
                    )[0]
        return f

    def scan_to (self, x, y, dx, dy):
        """
        Change qcut, a value clipped to in range 0 to 1 
        that is used for glDrawElements index clipping 
        (partial VBO drawing)

        This can for example be used for an interactive time slider
        """
        self.qcut += self.qcut*dy
        #log.info("DAEevent.scan_to %s " % repr(self)) 

    def time_to(self, x, y, dx, dy):
        """
        Use for real propagation time control, not the fake time of initial photon 
        variety.
        """
        self.dphotons.time_to( x, y, dx, dy )


    def reconfig(self, event_config ):
        """
        Handle argument sequences like::
        """ 
        photons_config = []
        load_config = {}

        key = None
        for k,v in event_config:
            if k == 'key':
                key = v
            elif k == 'save':
                self.save(v)
            elif k in ('load','type','slice'):
                load_config[k] = v 
            elif k == 'clear':
                self.clear()
            elif k == 'tcut':
                self.qcut = v 
            elif k == 'reload':
                self.reload_()
            elif k in ('fpholine','fphopoint','mode','mask','time','style',):   
                assert 0, "should be handled directly %s " % k 
                photons_config.append([k,v])
            else:
                assert 0, (k,v)
            pass
        pass
        if len(load_config)>0:
            self.load(load_config['load'], load_config)

        if len(photons_config)>0:
            self.dphotons.reconfig(photons_config)
        pass 

    def external_npy(self, npy ):
        if self.config.args.saveall:
            log.info("external_npy timestamp_save due to --saveall option")
            name = None  # None signals timestamp
            typ = None
            assert 0, "needs attention" 
            #self.config.save_npy( npl, name, npl.typ )   
        else:
            log.info("external_npy not saving ")
        pass
        self.setup_npy(npy) 
 
    def setup_npy(self, npy):
        """
        :param npl: NPY array, shape (nphoton,4,4) for photons

        This is invoked by:

        * `external_npy` when arriving over network
        * `load` when loading from file

        """
        assert len(npy.shape) == 3 , "unexpected npy.shape %s " % repr(shape)

        self.scene.chroma.incoming(npy)
        typ = NPY.detect_type(npy)
        log.info("incoming array detect_type: %s %s " % (typ, repr(npy.shape))) 

        if typ == "photon": 
            photons = Photon.from_array(npy)   
            self.setup_photons( photons ) 
        elif typ == "cerenkov" or typ == "scintillation": 
            genstep = G4Step.from_array(npy)
            self.setup_genstep(genstep)
        else:
            log.info("received NPY array of unhandled type %s %s " % (typ, repr(npy.shape)))
        pass

    def setup_genstep(self, genstep):
        self.dgenstep.array = genstep

    def setup_photons(self, photons ):
        """
        :param photons: NPY Photon instance 

        #. setting the photons property invalidates dependents like `.mesh`
           and subsequent access will recreate them 

        """
        self.dphotons.array = photons

        mesh = self.dphotons.mesh
        self.scene.bookmarks.create_for_object( mesh, 9 )
        self.objects = [mesh]

    def clear(self):
        log.info("clear setting photons to None")
        self.dphotons.photons = None


    def step(self, dcc):
        """
        Use Chroma propagation to step the photons. Note the 
        replacement of dphotons.photons arising from the 
        step and resultant VBO recreation.

        :param chroma: DAEChromaContext instance
        """
        assert 0 
        if self.dphotons is None:
            log.warn("cannot step without loaded dphotons")
            return
        pass
        log.info("step")

        photons = dcc.propagator.propagate( self.dphotons.photons, max_steps=1 )
        #photons.dump()
        self.setup_photons( photons )   # results in VBO recreation 

    def find_object(self, ospec):
        try:
            index = int(ospec)
        except ValueError:
            return None
        try:
            return self.objects[index]    
        except IndexError:
            return None

    def draw(self):
        if not self.dphotons is None:
            self.dphotons.draw()
        if not self.dgenstep is None:
            self.dgenstep.draw()
        pass

    #def save(self, path_, key=None ):
    #    if self.cpl is None:
    #        log.warn("no cpl, nothing to save ") 
    #        return
    #    self.config.save_cpl( path_, key, self.cpl.cpl )   
        
    def load(self, name, cfg={}):
        typ = cfg.get('type', self.config.args.type)
        sli = cfg.get('slice', self.config.args.slice)
        path = self.config.resolve_templated_path(name, typ)
        npy = self.config.load_npy( path, typ, sli)
        self.setup_npy( npy )

        if npy is None:
            log.warn("load of typ %s name %s path %s failed " % (typ,name,path))
            return
        pass
        self.loaded = path
        self.eventlist.path = path   # let eventlist know where we are, to allow loadnext loadprev
        self.config.rmenu.dispatch('on_needs_redraw')

    def reload_(self):
        path = self.eventlist.path 
        if not path is None:
            log.info("reload_ %s " % path )
            self.load(path) 
        else:
            log.warn("cannot reload as no current path")

    def loadnext(self):
        log.debug("loadnext")
        next_ = self.eventlist.next_  # using next_ bumps the cursor forwards
        if not next_ is None:
            self.load(next_) 

    def loadprev(self):
        log.debug("loadprev")
        prev = self.eventlist.prev  # using prev bumps the cursor backwards
        if not prev is None:
            self.load(prev) 




if __name__ == '__main__':
    pass


