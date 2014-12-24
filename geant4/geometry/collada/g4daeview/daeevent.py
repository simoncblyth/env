#!/usr/bin/env python

import logging, datetime
log = logging.getLogger(__name__)
import numpy as np

from daephotons import DAEPhotons
from daeeventlist import DAEEventList , DAEEventListMenu
from daemenu import DAEMenu
from daeanimator import DAEAnimator
from daeeventbase import DAEEventBase

class DAEEventMenu(DAEMenu):
    def __init__(self, config, handler):
        DAEMenu.__init__(self, "event")
        self.add("reload",handler.reload_)
        self.add("loadnext",handler.loadnext)
        self.add("loadprev",handler.loadprev)

class DAEEvent(DAEEventBase):
    """
    TODO: split this up further, doing too much 
    """
    def __init__(self, config, scene ):
        DAEEventBase.__init__(self, config, scene)

        self.loaded = None
        pass
        self.qcut = config.args.qcut 
        self.bbox_cache = None
        photons = None

        log.info("********* scene.event.dphotons ")
        self.dphotons = DAEPhotons( photons, self )

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
        self.dphotons.deferred_menu_update()

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
        return self.dphotons.time
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

            --key CPL --load /tmp/1.root --key OBJ --load /tmp/2.root 

        """ 
        key = self.config.args.key

        photons_config = []

        for k,v in event_config:
            if k == 'key':
                key = v
            elif k == 'save':
                self.save(v, key)
            elif k == 'load':
                self.load(v, key)
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
        if len(photons_config)>0:
            self.dphotons.reconfig(photons_config)
        pass 

    def external_cpl(self, cpl ):
        self.external_cpl_base( cpl )
    def external_npl(self, npl ):
        self.external_npl_base( npl )


    def setup_photons(self, photons ):
        """
        :param photons: `chroma.event.Photons` instance (or fallback)

        Slot the operations level chroma.event.Photons into the 
        DAEPhotons presentation controller instance.

        #. setting the photons property invalidates dependents like `.mesh`
           and subsequent access will recreate them 

        """
        self.setup_photons_base( photons )

        mesh = self.dphotons.mesh
        self.scene.bookmarks.create_for_object( mesh, 9 )
        self.objects = [mesh]

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
        if self.dphotons is None:return
        self.dphotons.draw()

    def save(self, path_, key=None ):
        if self.cpl is None:
            log.warn("no cpl, nothing to save ") 
            return
        self.config.save_cpl( path_, key, self.cpl.cpl )   

        
    def load(self, path_, key=None ):
        path = self.config.resolve_event_path( path_ )

        lpho = None
        if path[-4:] == ".npy":
            lpho = self.config.load_npl( path, key )
            self.setup_npl( lpho )
        elif path[-5:] == ".root":
            lpho = self.config.load_cpl( path, key )
            self.setup_cpl( lpho )
        else:
            log.warn("unexpected path extension %s ", path)
        pass          
        if lpho is None:
            log.warn("load failed ")
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


