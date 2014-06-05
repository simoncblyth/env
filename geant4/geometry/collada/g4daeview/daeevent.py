#!/usr/bin/env python

import logging
log = logging.getLogger(__name__)
import numpy as np

from env.chroma.ChromaPhotonList.cpl import load_cpl, save_cpl   # ROOT transport level 

#from photons import Photons                                      # numpy operations level 


#TODO: allow non-chroma nodes to load CPL Photons too
try:
    from chroma.event import Photons
except ImportError:
    Photons = None


from daephotons import DAEPhotons, DAEPhotonsMenu                                # OpenGL presentation level

from datetime import datetime
from daeeventlist import DAEEventList , DAEEventListMenu
from daemenu import DAEMenu

def timestamp():
    return datetime.now().strftime("%Y%m%d-%H%M%S")


class DAEEventMenu(DAEMenu):
    def __init__(self, config, handler):
        DAEMenu.__init__(self, "event")
        self.add("reload",handler.reload_)
        self.add("loadnext",handler.loadnext)
        self.add("loadprev",handler.loadprev)


class DAEEvent(object):
    def __init__(self, config, scene ):
        self.config = config
        self.scene = scene
        pass
        self.qcut = config.args.tcut 
        self.bbox_cache = None
        self.dphotons = None
        self.objects = []
        self.eventlist = DAEEventList(config.args.path_template)

        # dont like this menu setup here 

        event_menu = DAEEventMenu( config, self )
        self.config.rmenu.addSubMenu(event_menu) 

        eventlist_menu = DAEEventListMenu(self.eventlist, self.eventlist_callback )
        self.config.rmenu.addSubMenu(eventlist_menu) 

        photons_menu = DAEPhotonsMenu( config )
        self.config.rmenu.addSubMenu(photons_menu)
        self.photons_menu = photons_menu 

        self.apply_launch_config()  # maybe better done externally 


    def apply_launch_config(self):
        launch_config = [] 
        if not self.config.args.load is None:
            launch_config.append( ['load',self.config.args.load])
        if not self.config.args.key is None:
            launch_config.append( ['key',self.config.args.key])
        pass
        if len(launch_config) > 0:
            self.reconfig(launch_config)
        pass

    def eventlist_callback(self, item):
        path = item.extra['path'] 
        self.load(path)

    def __repr__(self):
        return "%5.2f" % self._qcut
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
            elif k == 'tcut':
                self.qcut = v 
            elif k == 'reload':
                self.reload_()
            elif k in ('fpholine','pholine','fphopoint','phopoint','mask'):   
                photons_config.append([k,v])
            else:
                assert 0, (k,v)
            pass
        pass
        if len(photons_config)>0:
            self.dphotons.reconfig(photons_config)
        pass 

    def external_cpl(self, cpl ):
        """
        :param cpl: ChromaPhotonList instance

        External ZMQ messages containing CPL arrive at DAEResponder and are routed here
        via glumpy event system.
        """
        if self.config.args.saveall:
            log.info("external_cpl timestamp_save due to --saveall option")
            self.timestamped_save(cpl)
        else:
            log.info("external_cpl not saving ")
        pass
        self.setup_cpl(cpl) 

    def timestamped_save(self, cpl):
        path_ = timestamp()
        path = self.resolve(path_, self.config.args.path_template)
        key = self.config.args.key 
        save_cpl( path, key, cpl )   
 
    def setup_cpl(self, cpl):
        """
        :param cpl: ChromaPhotonList instance

        Convert serialization level ChromaPhotonList into operation level Photons
        """
        if Photons is None:
            log.warn("setup_cpl requires chroma ")
            return

        photons = Photons.from_cpl(cpl, extend=True)   
        self.setup_photons( photons ) 

    def setup_photons(self, photons ):
        """
        Convert operations level Photons into presentation level DAEPhotons 
        """
        if self.dphotons is None:
            self.dphotons = DAEPhotons( photons, self )
            self.photons_menu.update_flags_menu()
        else:
            self.dphotons.photons = photons   # setter invalidates _vbo, _color, _mesh 
        pass

        mesh = self.dphotons.mesh
        #log.info("setup_photons mesh\n%s\n" % str(mesh))
        self.scene.bookmarks.create_for_object( mesh, 9 )
        self.objects = [mesh]

    def step(self, dcc):
        """
        :param chroma: DAEChromaContext instance
        Use Chroma propagation to step the photons 
        """
        if self.dphotons is None:
            log.warn("cannot step without loaded dphotons")
            return
        pass
        log.info("step")

        propagator = dcc.propagator
        photons = propagator.propagate( self.dphotons.photons, max_steps=1 )
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
        if key is None:
            key = self.config.args.key
        if self.cpl is None:
            log.warn("no cpl, nothing to save ") 
            return
        path = self.config.resolve_event_path(path_)
        log.info("save cpl into  %s : %s " % (path_, path) )
        save_cpl( path, key, self.cpl.cpl )   

    def load(self, path_, key=None ):
        if key is None:
            key = self.config.args.key
        path = self.config.resolve_event_path(path_)
        log.info("load cpl from  %s : %s " % (path_, path) )
        cpl = load_cpl(path, key )
        if cpl is None:
            log.warn("load_cpl failed ")
            return
        pass
        self.eventlist.path = path   # let eventlist know where we are, to allow loadnext loadprev
        self.setup_cpl( cpl )
        self.config.rmenu.dispatch('on_needs_redraw')

    def reload_(self):
        path = self.eventlist.path 
        if not path is None:
            log.info("reload_ %s " % path )
            self.load(path) 
        else:
            log.warn("cannot reload as no current path")

    def loadnext(self):
        log.info("loadnext")
        next_ = self.eventlist.next_  # using next_ bumps the cursor forwards
        if not next_ is None:
            self.load(next_) 

    def loadprev(self):
        log.info("loadprev")
        prev = self.eventlist.prev  # using prev bumps the cursor backwards
        if not prev is None:
            self.load(prev) 




if __name__ == '__main__':
    pass


