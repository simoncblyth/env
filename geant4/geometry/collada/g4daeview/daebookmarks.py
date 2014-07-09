#!/usr/bin/env python
import os, logging
log = logging.getLogger(__name__)

from ConfigParser import ConfigParser
from daeinterpolateview import DAEInterpolateView, DAEParametricView
from daeviewpoint import DAEViewpoint
from daeclipper import DAEClipper
from daecamera import DAECamera

class DAEBookmarks(object):
    name = "bookmarks.cfg"
    path = property(lambda self:self.config.resolve_confpath(self.name))

    def __init__(self, config, geometry ):
        """
        :param config: to bookmarks file
        :param geometry: DAEGeometry instance
        """
        self.config = config
        self.current = None

        self.viewpoints = {}
        self.clippers = {} 
        self.cameras = {} 

        if os.path.exists(self.path):
            self.load(geometry)  

    ini_prefix = "bookmark_"
    ini_exclude = ("0",)

    def _get_marks(self):
        """
        :return: list of bookmark keys k with associated .viewpoints[k].solid excluding k=0 
        """
        marks = filter(lambda k:k not in self.ini_exclude, sorted(self.viewpoints))  # bookmark_0 excluded
        marks = filter(lambda k:not self.viewpoints[k].solid is None,marks)  
        return marks
    marks = property(_get_marks,doc=_get_marks.__doc__)

    def _get_asini(self):
        """
        :return: ini format string encoding view and clipping config sections for each bookmark 
        """
        entries = []
        for k in self.marks:
            key_hdr = "[%s%s]" % (self.ini_prefix, k)
            entries.append(key_hdr)
            entries.append(self.viewpoints[k].asini)
            entries.append(self.cameras[k].asini)
            entries.append(self.clippers[k].asini)
            pass 
        return "\n".join(entries)
    asini = property(_get_asini, doc=_get_asini.__doc__)

    def _get_clipper(self):
        """
        Access clipper corresponding to the current bookmark, creates the instance 
        if does not already exist.
        """
        if not self.current in self.clippers:
            self.clippers[self.current] = DAEClipper()  
        return self.clippers[self.current] 
    clipper = property(_get_clipper, doc=_get_clipper.__doc__)

    def _get_camera(self):
        """
        Access camera corresponding to the current bookmark, creates the default instance 
        from config if does not already exist.
        """
        if not self.current in self.cameras:
            self.cameras[self.current] = DAECamera.fromconfig(self.config)
        return self.cameras[self.current] 
    camera = property(_get_camera, doc=_get_camera.__doc__)


    def add_clipping_plane(self, plane):
        """
        Add the plane to the clipper corresponding to the current bookmark
        """ 
        log.info("add_clipping_plane")
        self.clipper.add(plane)

    def save(self):
        """
        Saves .asini to .path
        """
        dir_ = os.path.dirname(self.path)
        if not os.path.exists(dir_):
            log.info("creating directory %s " % dir_ )
            os.makedirs(dir_) 
        pass
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
                    continue

                self.assign(k, view)
                self.clippers[k] = DAEClipper.from_ini(cfg)
                self.cameras[k] = DAECamera.fromini(cfg)

    def __repr__(self):
        def fmt_(k):
            return "[%s]" if self.is_current(k) else "%s"  
        return "".join(map(lambda k:fmt_(k) % k,sorted(self.viewpoints.keys())))

    def create_for_solid(self, solid, numkey):
        log.debug("create_for_solid: numkey %s solid.id %s" % (numkey,solid.id) )
        view = self.transform.spawn_view_jumping_frame(solid)
        self.assign(numkey, view)
        self.set_current(numkey)

    def create_for_object(self, obj, numkey):
        #log.info("create_for_object: numkey %s " % (numkey) )
        view = self.transform.spawn_view_jumping_frame(obj)
        self.assign(numkey, view)
        self.set_current(numkey)

    def assign(self, key, view):
        """
        Record the `view` as bookmark `key` into this dict

        Whats an appropriate relationship between clips(clipping planes), views and bookmarks ?
        Views come and go, so not a good place to stick the clips.

        :param key:
        :param view:
        """
        self.viewpoints[str(key)] = view

    def lookup(self, key, default=None):
        return self.viewpoints.get(str(key),default)
    def set_current(self, key):
        self.current = str(key)  
    def is_current(self, key):
        return str(key) == self.current
 
    current_view = property(lambda self:self.lookup(self.current,None))

    def update_current(self):
        """
        
        """
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
        oldkey = self.current
        if oldkey in self.clippers:
            log.info("disable oldkey %s clips " % oldkey )
            self.clippers[oldkey].disable()
        else:
            log.warn("huh no oldkey %s in clippers " % oldkey ) 
        pass

        view = self.lookup(numkey, None)
        if not view is None:
            self.set_current(numkey)
        return view

    def make_interpolate_view(self):
        """
        """
        if len(self) < 2:
            return None
        pass
        keys = sorted(self, key=lambda _:_)
        idx = keys.index(self.current)
        keys_starting_with_current = keys[idx:] + keys[:idx] 
        views = [self.viewpoints[k] for k in keys_starting_with_current]
        log.info("make_interpolate_view sequence with %s views " % (len(views)))
        return DAEInterpolateView(views)

    def make_parametric_view(self):
        log.info("make_parametric_view for current view %s " % (self.current))
        return DAEParametricView(self.current_view)
if __name__ == '__main__':
    pass


