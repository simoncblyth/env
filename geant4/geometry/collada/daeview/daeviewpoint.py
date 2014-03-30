#!/usr/bin/env python

import logging
log = logging.getLogger(__name__)
import numpy as np
from daeutil import printoptions, WorldToCamera, CameraToWorld



class DAEViewpoint(object):
    """
    Changes to model frame _eye/_look/_up made 
    after instantiation are immediately reflected in 
    the results obtained from the output properties: eye, look, up, eye_look_up

    The transform and its extent are however fixed.
    """
    interpolate = False
    def __init__(self, eye, look, up, solid, target ):
        """
        :param eye: model frame camera position, typically (1,1,0) or similar
        :param look: model frame object that camera is pointed at, usually (0,0,0)
        :param up: model frame up, often (0,0,1)
        :param model2world: transform from model frame to world frame
        :param world2model: opposite transfrom back from world frame back into model frame
        :param target: string identifier for the corresponding solid
        """
        self._eye = np.array(eye) 
        self._look = np.array(look)  
        self._up = np.array(up)     # a direction 
        pass
        self.model2world = solid.model2world
        self.world2model = solid.world2model
        self.world2camera = WorldToCamera( self.eye, self.look, self.up )
        self.camera2world = CameraToWorld( self.eye, self.look, self.up )
        self.extent = solid.extent  
        self.target = target
        self.index = solid.index
        self.solid = solid
        pass

    def __call__(self, f):
        log.warn("not an interpolatable view ")

    views = property(lambda self:[self])  # mimic DAEInterpolateView 
    current_view = property(lambda self:self)  
    next_view = property(lambda self:None)      


    eye  = property(lambda self:self.model2world(self._eye))
    look = property(lambda self:self.model2world(self._look))
    up   = property(lambda self:self.model2world(self._up,w=0.))
    

    def _get_distance(self):
        model2world = self.model2world
        eye = model2world(self._eye)
        look = model2world(self._look)
        gaze = look - eye
        return np.linalg.norm(gaze) 
    distance = property(_get_distance)

    def _get_eye_look_up(self):
        """
        Provides eye,look,up in world frame coordinates
        """
        model2world = self.model2world
        eye = model2world(self._eye)
        look = model2world(self._look)
        up = model2world(self._up,w=0)
        return np.concatenate([eye[:3],look[:3],up[:3]])
    eye_look_up = property(_get_eye_look_up)

    def __repr__(self):
        """
        Express vecs in the shortest form, similar to human input style
        """
        fmt = "%.2f"
        isint_ = lambda _:np.allclose(float(int(_)),_)
        def nfmt(_):
            if isint_(_): 
                return "%s" % int(_)
            else:
                return fmt % _
        def brief_(v):
            return ",".join(map(nfmt,v[0:3]))

        return "V %s/%s %.2f e %s l %s d %.2f" % (self.target, self.index, self.extent, brief_(self._eye), brief_(self._look), self.distance  )  



    def smry(self):
        eye, look, up = np.split(self.eye_look_up,3)
        with printoptions(precision=3, suppress=True, strip_zeros=False):
            return "\n".join([
                    "%s " % self.__class__.__name__,
                    "p_eye  %s eye  %s " % (eye,  self._eye),
                    "p_look %s look %s " % (look, self._look),
                    "p_up   %s up   %s " % (up,   self._up),
                     self.solid.smry(),
                      ])





def check_solid():
    from daegeometry import DAEGeometry
    dg = DAEGeometry("3153:12230")
    dg.flatten()

    solid = dg.solids[5000]
    print solid
    print solid.smry()

    print solid.world2model.matrix   
    print solid.model2world.matrix   

    for p in [solid.center] + list(solid.bounds):
        print "world point          ", p
        print "model point from w2m ", solid.world2model(p)



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    check_solid()


