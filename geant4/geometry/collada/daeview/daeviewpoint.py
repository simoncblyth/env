#!/usr/bin/env python
import logging, math
log = logging.getLogger(__name__)
import numpy as np
from daeutil import printoptions, Transform, translate_matrix, scale_matrix
from daeutil import WorldToCamera, CameraToWorld, view_transform


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
    :param svec: input parameter

    Allow specification of a prior vec element value with "." thus::
       
         2000_.,.,10   # means take a look from 10 units above current position  
         # no it doesnt this means, same xy but move to 10
       

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

        The solid defines the coordinate frame in which eye,look,up are expressed
        """
        _eye, _look, _up = ensure_not_collinear( _eye, _look, _up )
        pass  
        # model frame input parameters
        self._eye = _eye
        self._look = _look  
        self._up = _up   # a direction 

        pass
        # the below are fixed for the view, cannot change the solid associated with a Viewpoint
        self.solid = solid
        self.target = target  # informational
        pass

    # NB the input parameters to the transforms are world coordinates
    index = property(lambda self:self.solid.index)
    extent = property(lambda self:self.solid.extent)
    model2world = property(lambda self:self.solid.model2world)
    world2model = property(lambda self:self.solid.world2model)


    #world2camera = property(lambda self:WorldToCamera( self.eye, self.look, self.up ))
    #camera2world = property(lambda self:CameraToWorld( self.eye, self.look, self.up ))

    def _get_world2camera(self): 
        return view_transform( self.eye, self.look, self.up, inverse=False ) 
    world2camera = property(_get_world2camera)

    def _get_camera2world(self): 
        return view_transform( self.eye, self.look, self.up, inverse=True ) 
    camera2world = property(_get_camera2world)



    def change_solid(self, solid ):
        """
        :param solid: DAESolid instance

        Changing solid is a drastic action for a DAEViewpoint, it means a jump
        of frames of reference from the old to the new solid entailing a change
        to every item of state of the viewpoint, while retaining the same 
        visual impression.
        """
        pass
        raise Exception("using transform from Viewpoint would be confusing")


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
   
        velem = str(vspec).split("_") if not vspec is None else [] 

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


    def _get_translate_eye2look(self):
        """eye frame translation from eye to look position"""
        return translate_matrix((0,0,   self.distance)) 
    translate_eye2look = property(_get_translate_eye2look)

    def _get_translate_look2eye(self):
        """eye frame translation from look to eye position"""
        return translate_matrix((0,0,  -self.distance)) 
    translate_look2eye = property(_get_translate_look2eye)

    def _get_distance(self):   # hmm could use invariant here 
        model2world = self.model2world
        eye = model2world(self._eye)
        look = model2world(self._look)
        gaze = look - eye
        return np.linalg.norm(gaze) 
    distance = property(_get_distance, doc="distance from eye to look" )

    def _get_eye_look_up(self):
        """
        NB this is viewpoint initial eye, look, up not the dynamically updating one as trackball around
        """
        model2world = self.model2world
        eye = model2world(self._eye)
        look = model2world(self._look)
        up = model2world(self._up,w=0)
        return np.concatenate([eye[:3],look[:3],up[:3]])
    eye_look_up = property(_get_eye_look_up, doc="9 element array containing eye, look, up in world frame coordinates" )

    def _get_eye_look_up_model(self):
        """
        NB this is viewpoint initial eye, look, up not the dynamically updating one as trackball around
        """
        return np.vstack([np.append(self._eye,1),np.append(self._look,1),np.append(self._up,0)])
    eye_look_up_model = property(_get_eye_look_up_model, doc="3x4 element array containing eye, look, up in homogenous model frame coordinates" )

    def _get_eye_look_up_world(self):
        return self.model2world.matrix.dot( self.eye_look_up_model.T ).T 
    eye_look_up_world = property(_get_eye_look_up_world, doc="3x4 element array containing eye, look, up in homogenous world frame coordinates" )


    def elu(self):
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
        return "e %s l %s d %.2f" % (brief_(self._eye), brief_(self._look), self.distance  )  

    def __repr__(self):
        return "V %s/%s x %.2f d %.2f" % (self.target, self.index, self.extent, self.distance  )  

    def __str__(self):
        eye, look, up = np.split(self.eye_look_up,3)
        with printoptions(precision=3, suppress=True, strip_zeros=False):
            return "\n".join([
                    "%s (non-trackballed world elu)" % self.__class__.__name__,
                    "eye  %s _eye  %s " % (eye,  self._eye),
                    "look %s _look %s " % (look, self._look),
                    "up   %s _up   %s " % (up,   self._up),
            #         self.solid.smry(),
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
    view = DAEViewpoint( (-2,0,0), (0,0,0), (0,0,1), solid, "")  
    print view
    print view.eye_look_up_model
    print view.eye_look_up_world





 
