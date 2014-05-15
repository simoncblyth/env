#!/usr/bin/env python

import logging
log = logging.getLogger(__name__)
import numpy as np
from daechromaphotonlistbase import DAEChromaPhotonListBase
from daechromaphotonlist import DAEChromaPhotonList
from daegeometry import DAEMesh 
from env.chroma.ChromaPhotonList.cpl import load_cpl, save_cpl   # uses ROOT
from datetime import datetime
from daeeventlist import DAEEventList 

def timestamp():
    return datetime.now().strftime("%Y%m%d-%H%M%S")


class DAEEvent(object):
    def __init__(self, config ):
        self.config = config
        self.cpl = None 
        self.qcut = config.args.tcut 
        self.eventlist = DAEEventList(config.args.path_template)
        self.bbox_cache = None
        self.objects = []

        launch_config = [] 
        if not self.config.args.load is None:
            launch_config.append( ['load',self.config.args.load])
        if not self.config.args.key is None:
            launch_config.append( ['key',self.config.args.key])
        pass
        if len(launch_config) > 0:
            self.reconfig(launch_config)
        pass

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

        cpl_config = []

        for k,v in event_config:
            if k == 'key':
                key = v
            elif k == 'save':
                self.save(v, key)
            elif k == 'load':
                self.load(v, key)
            elif k == 'tcut':
                self.qcut = v 
            elif k in ('fpholine','pholine','fphopoint','phopoint'):   
                cpl_config.append([k,v])
            else:
                assert 0, (k,v)
            pass
        pass
        if len(cpl_config)>0:
            self.cpl.reconfig(cpl_config)
        pass 

    def external_cpl(self, cpl ):
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
        dcpl = DAEChromaPhotonList(cpl, self, timesort=True, chroma=self.config.args.with_chroma)
        self.cpl = dcpl
        mesh = DAEMesh(self.cpl.pos)
        log.info("setup_cpl mesh\n%s\n" % str(mesh))
        self.objects = [mesh]

    def step(self, chroma_ctx):
        if self.cpl is None:
            log.warn("cannot step without loaded CPL")
            return
        log.info("step")
        photons2 = chroma_ctx.step( self.cpl )
        DAEChromaPhotonListBase.dump_(photons2) 



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
        if self.cpl is None:return
        self.cpl.draw()

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


