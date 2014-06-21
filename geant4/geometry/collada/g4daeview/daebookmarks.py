#!/usr/bin/env python
import os, logging
log = logging.getLogger(__name__)

from ConfigParser import ConfigParser
from daeinterpolateview import DAEInterpolateView, DAEParametricView
from daeviewpoint import DAEViewpoint

class DAEBookmarks(dict):
    def __init__(self, path, geometry):
        dict.__init__(self)
        self.path = path
        self.current = None
        if os.path.exists(path):
            self.load(geometry)  

    ini_prefix = "bookmark_"
    ini_exclude = ("0",)

    def _get_marks(self):
        marks = filter(lambda k:k not in self.ini_exclude, sorted(self))  # bookmark_0 excluded
        marks = filter(lambda k:not self[k].solid is None,marks)  
        return marks
    marks = property(_get_marks)

    def _get_asini(self):
        return "\n".join(["[%s%s]\n%s" % (self.ini_prefix, k, self[k].asini) for k in self.marks])
    asini = property(_get_asini)

    def save(self):
        log.info("save %s bookmarks to %s " % (len(self.marks),self.path ))
        with open(self.path,"w") as w:
            w.write(self.asini + "\n")

    def load(self, geometry):
        """
        :param geometry: DAEGeometry instance
        """
        log.debug("load bookmarks from %s " % self.path )
        cfp = ConfigParser()
        cfp.read([self.path])        

        for sect in cfp.sections():
            if sect.startswith(self.ini_prefix):
                k = sect[len(self.ini_prefix):]
                cfg = cfp.items(sect)
                view = DAEViewpoint.fromini( cfg, geometry ) 
                if view is None:
                    log.debug("failed to load bookmark %s " % k )
                else:   
                    self.assign(k, view)

    def __repr__(self):
        def fmt_(k):
            return "[%s]" if self.is_current(k) else "%s"  
        return "".join(map(lambda k:fmt_(k) % k,sorted(self.keys())))

    def create_for_solid(self, solid, numkey ):
        log.info("create_for_solid: numkey %s solid.id %s" % (numkey,solid.id) )
        view = self.transform.spawn_view_jumping_frame(solid)
        self.assign(numkey, view)
        self.set_current(numkey)

    def create_for_object(self, obj, numkey ):
        log.info("create_for_object: numkey %s " % (numkey) )
        view = self.transform.spawn_view_jumping_frame(obj)
        self.assign(numkey, view)
        self.set_current(numkey)

    def assign(self, key, view):
        self[str(key)] = view
    def lookup(self, key, default=None):
        return self.get(str(key),default)
    def set_current(self, key):
        self.current = str(key)  
    def is_current(self, key):
        return str(key) == self.current
 
    current_view = property(lambda self:self.lookup(self.current,None))

    def update_current(self):
        numkey = self.current
        if numkey is None:
            log.warn("no current bookmark")
            return
        view = self.lookup(numkey, None)
        if view is None:
            log.warn("no such bookmark %s cannot update " % numkey )
            return  
        log.info("updating bookmark %s view.solid.id %s " % (numkey, view.solid.id))
        view = self.transform.spawn_view_jumping_frame(view.solid)
        self.assign(numkey, view) 

    def visit(self, numkey):
        view = self.lookup(numkey, None)
        if not view is None:
            self.set_current(numkey)
        return view

    def make_interpolate_view(self):
        if len(self) < 2:
            return None
        pass
        keys = sorted(self, key=lambda _:_)
        idx = keys.index(self.current)
        keys_starting_with_current = keys[idx:] + keys[:idx] 
        views = [self[k] for k in keys_starting_with_current]
        log.info("make_interpolate_view sequence with %s views " % (len(views)))
        return DAEInterpolateView(views)

    def make_parametric_view(self):
        log.info("make_parametric_view for current view %s " % (self.current))
        return DAEParametricView(self.current_view)
if __name__ == '__main__':
    pass


