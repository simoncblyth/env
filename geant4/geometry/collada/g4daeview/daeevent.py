#!/usr/bin/env python

import logging
log = logging.getLogger(__name__)
import numpy as np
from daechromaphotonlist import DAEChromaPhotonList
from env.chroma.ChromaPhotonList.cpl import load_cpl, save_cpl   # uses ROOT


class DAEEvent(object):
    def __init__(self, config ):
        self.config = config
        self.cpl = None 
        self._qcut = 1. 

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
        self._qcut = np.clip(qcut, 0., 1.)
    qcut = property(_get_qcut, _set_qcut)


    def scan_to (self, x, y, dx, dy):
        """
        Change qcut, a value clipped to in range 0 to 1 
        that is used for glDrawElements index clipping 
        (partial VBO drawing)

        This can for example be used for an interactive time slider
        """
        self.qcut += self.qcut*dy
        log.info("DAEevent.scan_to %s " % repr(self)) 

    def reconfig(self, event_config ):
        """
        Handle argument sequences like::

            --key CPL --load /tmp/1.root --key OBJ --load /tmp/2.root 

        """ 
        key = self.config.args.key
        for k,v in event_config:
            if k == 'key':
                key = v
            elif k == 'save':
                self.save(v, key)
            elif k == 'load':
                self.load(v, key)
            else:
                assert 0, (k,v)
            pass
        pass

    def external_cpl(self, cpl ):
        log.info("external_cpl")
        cpl = DAEChromaPhotonList(cpl, self)
        self.cpl = cpl

    def draw(self):
        if self.cpl is None:return
        self.cpl.draw()

    @classmethod
    def resolve(cls, path_, path_template ):
        """
        Using a path_template allows referencing paths in a
        very brief manner, ie with::
 
            export DAE_PATH_TEMPLATE="/usr/local/env/tmp/%(arg)s.root"

        Can use args `--load 1` 

        """
        if path_template is None:
            return path_
        log.info("resolve path_template %s path_ %s " % (path_template, path_ )) 
        path = path_template % { 'arg':path_ }
        return path 

    def save(self, path_, key ):
        path = cls.resolve(path_, self.config.args.path_template)
        if self.cpl is None:
            log.warn("no cpl, nothing to save ") 
            return
        pass
        log.info("save cpl into  %s : %s " % (path_, path) )
        save_cpl( path, key, self.cpl.cpl )   

    def load(self, path_, key ):
        path = self.resolve(path_, self.config.args.path_template)
        log.info("load cpl from  %s : %s " % (path_, path) )
        cpl = load_cpl(path, key )
        if cpl is None:
            log.warn("load_cpl failed ")
            return
        pass
        self.external_cpl( cpl )


if __name__ == '__main__':
    pass


