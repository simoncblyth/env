#!/usr/bin/env python
import os, logging
log = logging.getLogger(__name__)
from daeinterpolateview import DAEInterpolateView, DAEParametricView

class DAEBookmarks(dict):
    def __init__(self):
        dict.__init__(self)
        self.current = None

    def __repr__(self):
        return "".join(map(str,self.keys()))

    def create_for_solid(self, solid, numkey ):
        log.info("create_for_solid: numkey %s solid.id %s" % (numkey,solid.id) )
        view = self.transform.spawn_view_jumping_frame(solid)
        self[numkey] = view
        self.current = numkey

    current_view = property(lambda self:self.get(self.current,None))

    def update_current(self):
        numkey = self.current
        if numkey is None:
            log.warn("no current bookmark")
            return
        view = self.get(numkey, None)
        if view is None:
            log.warn("no such bookmark %s cannot update " % numkey )
            return  
        log.info("updating bookmark %s view.solid.id %s " % (numkey, view.solid.id))
        self[numkey] = self.transform.spawn_view_jumping_frame(view.solid)


    def visit(self, numkey):
        view = self.get(numkey, None)
        if not view is None:
            self.current = numkey  
        return view

    def make_interpolate_view(self):
        views = [self[k] for k in sorted(self,key=lambda _:_)]
        log.info("make_interpolate_view sequence with %s views " % (len(views)))
        return DAEInterpolateView(views)

    def make_parametric_view(self):
        log.info("make_parametric_view for current view %s " % (self.current))
        return DAEParametricView(self.current_view)



if __name__ == '__main__':
    pass


