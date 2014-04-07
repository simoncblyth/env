#!/usr/bin/env python

import logging, math
log = logging.getLogger(__name__)
import numpy as np
from daeutil import printoptions, WorldToCamera, CameraToWorld, Transform


def rotate( th , axis="x"):
    c = math.cos(th)
    s = math.sin(th)
    if axis=="x":
        m = np.array([[1,0,0],[0,c,-s],[0,s,c]])
    elif axis=="y":
        m = np.array([[c,0,s],[0,1,0],[-s,0,c]])
    elif axis=="z":
        m = np.array([[c,-s,0],[s,c,0],[0,0,1]])
    else:
        assert 0
    return m 

def ensure_not_collinear( eye, look, up):
    e = np.array(eye)
    l = np.array(look)
    g = l - e
    u = np.array(up)     
    if np.allclose(np.cross(g,u),(0,0,0)):
        m = rotate( 2.*math.pi/180. , "x")
        u = np.dot(m,u) 
        log.info("tweaking up vector as collinear vectors")

    return e,l,u

def pfvec_( svec, prior ):
    """
    Allow specification of a prior vec element value with "." thus::
       
    2000_.,.,10   # means take a look from 10 units above current position  

    """
    if svec is None:
        return prior
    float_or_prior_ = lambda _:sp[1] if sp[0] == "." else float(sp[0])
    return [float_or_prior_(sp) for sp in zip(svec.split(","),prior)]


class DAEViewpoint(object):
    """
    Changes to model frame _eye/_look/_up made 
    after instantiation are immediately reflected in 
    the results obtained from the output properties: eye, look, up, eye_look_up

    The transform and its extent are however fixed.
    """
    interpolate = False

    eye  = property(lambda self:self.model2world(self._eye))
    look = property(lambda self:self.model2world(self._look))
    up   = property(lambda self:self.model2world(self._up,w=0.))
 
    def __init__(self, _eye, _look, _up, solid, target ):
        """
        :param eye: model frame camera position, typically (1,1,0) or similar
        :param look: model frame object that camera is pointed at, usually (0,0,0)
        :param up: model frame up, often (0,0,1)
        :param solid: DAESolid(DAEMesh) instance, a set of fixed vertices (world frame)
        :param target: string identifier for the solid
        """
        _eye, _look, _up = ensure_not_collinear( _eye, _look, _up )
        pass  
        # model frame input parameters
        self._eye = _eye
        self._look = _look  
        self._up = _up   # a direction 

        pass
        # the below are fixed for the view, cannot change the solid associated with a Viewpoint
        self.model2world = solid.model2world
        self.world2model = solid.world2model

        self.extent = solid.extent  
        self.index = solid.index
        self.solid = solid
        self.target = target  # informational
        pass

    # NB the input parameters to the transforms are world coordinates
    world2camera = property(lambda self:WorldToCamera( self.eye, self.look, self.up ))
    camera2world = property(lambda self:CameraToWorld( self.eye, self.look, self.up ))


    def offset_eye_position(self, camera ):
        """
        :param camera: camera frame offset position, eg from trackball.xyz translation
        :return: model frame coordinates of offset eye position

        Original eye of the view is semi-fixed.
        Trackball translations do not change the view instance eye.

        TODO: debug this further, getting unexpected factor of two from somewhere 
        """
        world = self.camera2world( -camera )
        offset = self.world2model( world[:3] )   # model frame
        effective = self._eye + offset[:3]
        #log.info("whereami : camera %s world %s offset %s effective %s " % (camera, world, offset, effective  ) ) 
        return effective
 

    def change_eye_look_up(self, eye=None, look=None, up=None):
        """ 
        :param eye:  model frame "eye" position
        :param look: model frame "look" position
        :param up:   model frame "up" vector

        Model frame changes are immediately reflected in the world frame output properties

        TODO: check eye=look error handling
        """ 
        eye = pfvec_(eye, self._eye )
        look = pfvec_(look, self._look )
        up = pfvec_(up, self._up )

        _eye, _look, _up = ensure_not_collinear( eye, look, up )

        self._eye = _eye
        self._look = _look  
        self._up = _up  # a direction 


    @classmethod
    def interpret_vspec( cls, vspec, args ):
        """
        Interpret view specification strings like the below into
        target, eye, look, up::

           2000
           +0
           +1
           -1
           2000_0,1,1
           2000_-1,-1,-1
           2000_-1,-1,-1
           2000_-1,-1,-1_0,0,1
           2000_-1,-1,-1_0,0,1_0,0,1


        """
        fvec_ = lambda _:map(float, _.split(","))

        target = None
        eye = fvec_(args.eye)
        look = fvec_(args.look)
        up = fvec_(args.up)
   
        velem = vspec.split("_") if not vspec is None else [] 

        nelem = len(velem)
        if nelem > 0: 
            target = velem[0]  
        if nelem > 1:
            eye = pfvec_(velem[1], eye )
        if nelem > 2:
            look = pfvec_(velem[2], look )
        if nelem > 3:
            up = pfvec_(velem[3], up )

        return target, eye, look, up

    @classmethod
    def make_view(cls, geometry, vspec, args, prior=None ):
        """
        :param target: when None corresponds to the full mesh 

        The transform converts solid frame coordinates expressed in units of the extent
        into world frame coordinates.
        """
        target, eye, look, up = cls.interpret_vspec( vspec, args )
        if target == ".": 
            if prior is None: 
                target = ".."
                log.info("make_view:target spec of . but no prior.index fallback to entire mesh ")
            else:
                target = str(prior.index) 
                log.info("make_view:target spec of . interpreted as prior.index %s  " % target )
            pass
        elif target is None:
            log.info("make_view:target is None, defaulting to entire mesh ")
            target = ".."
        else:
            pass


        solid = geometry.find_solid(target) 
        if solid is None:
            log.warn("make_view failed to find solid for target %s : that node not loaded ? " % target )
            return None
        return cls(eye, look, up, solid, target )   


    def __call__(self, f, g):
        log.warn("not an interpolatable view ")

    views = property(lambda self:[self])  # mimic DAEInterpolateView 
    current_view = property(lambda self:self)  
    next_view = property(lambda self:None)      


    def _get_distance(self):
        model2world = self.model2world
        eye = model2world(self._eye)
        look = model2world(self._look)
        gaze = look - eye
        return np.linalg.norm(gaze) 
    distance = property(_get_distance, doc="distance from eye to look" )

    def _get_eye_look_up(self):
        model2world = self.model2world
        eye = model2world(self._eye)
        look = model2world(self._look)
        up = model2world(self._up,w=0)
        return np.concatenate([eye[:3],look[:3],up[:3]])
    eye_look_up = property(_get_eye_look_up, doc="9 element array containing eye, look, up in world frame coordinates" )

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


def check_view(geometry):
    target = "+0"
    eye = (1,1,1)
    look = (0,0,0)
    up = (0,1,0)
 
    view = DAEViewpoint.make_view( geometry, target, eye, look, up)
    print view 

    view._eye = np.array([1.1,1,1])
    print view 





class DummySolid(object):
    model2world = Transform()
    world2model = Transform()
    extent = 100.
    index = 1


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    #check_solid()

    from daecamera import DAECamera  
    camera = DAECamera()

    solid = DummySolid()
    view = DAEViewpoint( (0,2,0), (0,0,0), (0,0,1), solid, "")  
    print view

    pixel2camera = camera.pixel2camera
    camera2world = view.camera2world.matrix
    pixel2world = np.dot( camera2world, pixel2camera )

    corners = np.array(camera.pixel_corners.values())

    worlds  = np.dot( corners, pixel2world.T )
    worlds2 = np.dot( pixel2world, corners.T ).T   
    assert np.allclose( worlds, worlds2 )





 
