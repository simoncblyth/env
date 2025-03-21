#!/usr/bin/env python
"""
"""
import logging, math
from daeutil import WorldToCamera, CameraToWorld, view_transform

log = logging.getLogger(__name__)
import numpy as np

class DummyView(object):
    def __init__(self, j):
        self.j = j 
    def __repr__(self):
        return "%s %s " % (self.__class__.__name__, self.j )
    eye_look_up = property(lambda self:self.j*np.arange(1,10))

def oscillate( frame, low, high, speed ):
    return low + (high-low)*(math.sin(frame*math.pi*speed)+1.)/2. 

def sawtooth( frame, low, high, speed ):
    p = int(1./speed)
    return low + (high-low)* float(frame%(p+1))/float(p) 

   

class DAEViewpointBase(object):
    """
    Subclasses **MUST** define properties:

    #. current_view: DAEViewpoint instance
    #. eye_look_up: 9-element np.array providing world frame eye, look, up 

    * OpenGL feels the interpolation via view.eye_look_up which is passed to gluLookAt 

    * Chroma Raycaster only has transform.eye and transform.pixel2world which
      depends on view.camera2world thus the camera2world matrix must be
      dynamically updated by the interpolation

    """
    # pass along non-interpolated attribute access to current view, ie the "A" view for DAEInterpolatedView
    solid = property(lambda self:self.current_view.solid)
    extent = property(lambda self:self.current_view.extent)
    target = property(lambda self:self.current_view.target)
    index = property(lambda self:self.current_view.index)
    model2world = property(lambda self:self.current_view.model2world)
    world2model = property(lambda self:self.current_view.world2model)
    translate_look2eye = property(lambda self:self.current_view.translate_look2eye)
    translate_eye2look = property(lambda self:self.current_view.translate_eye2look)

    def _get_world2camera(self): 
        eye,look,up = np.split(self.eye_look_up,3)
        return view_transform( eye, look, up, inverse=False ) 
    world2camera = property(_get_world2camera)

    def _get_camera2world(self): 
        eye,look,up = np.split(self.eye_look_up,3)
        return view_transform( eye, look, up, inverse=True ) 
    camera2world = property(_get_camera2world)

    def _get_distance(self):
        eye,look,up = np.split(self.eye_look_up,3)
        return np.linalg.norm( look-eye)
    distance = property(_get_distance)

    eye  = property(lambda self:self.eye_look_up[0:3])
    look = property(lambda self:self.eye_look_up[3:6])
    up  =  property(lambda self:self.eye_look_up[6:9])

    def __repr__(self):
        return "%s" % self.__class__.__name__


class DAEParametricView(DAEViewpointBase):
    """
    Builds upon a basis view, that provides:
 
    #. a coordinate frame
    #. model frame state : _eye, _look, _up  (E, L, U)

    Consider a cylinder,

    #. axis direction/position given by L, U
    #. view distance EL defines radius

    ::


             U
             ^        | 
             |       Axis
             +--------|--------+ 
             |        |        |
             |        |        |
             |        |        |
             E........L........|
             |        |        |
             |        |        |
             |        |        |
             +-----------------+            
                      |

    Parametric view: e, l, u

        e    =   (  cos t,  sin t,  0 )   
        l    =   ( -sin t,  cos t,  0 )  # tangent look
        u    =   (      0,      0,  1 )  # orbiting around world frame Z only, for simplicity

    Look/rotation point overridden to correspond to eye point
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Trackballing whilst orbiting was almost impossible when the 
    look point about which rotation is made 
    is somehere off yonder... at a tangent.

    Override setting the view.distance to zero was not enough,
    needed to also override translate_look2eye/translate_eye2look 
    as transform was calling that on scene.view (ie this view) 
    which was then deferring to the basis view, which did not have distance 
    overriden. 

    After fixing that succeed to get Chroma raycast honoring the
    interpolated parametric viewpoint, and it is at least possible to 
    navigate now while parametrically orbiting a target.


    Hmm, looks like the rotate about self is not updating as trackball
    around

    """
    interpolate = True
    def __init__(self, view):
        """
        :param view: basis view 
        """
        DAEViewpointBase.__init__(self) 
        self.basis_view = view 
        self.current_view = view 

        x,y,z = view._eye[0:3]
        phase = np.arctan2(y,x)    # to prevent a jump when switching to orbting 

        log.info("basis_view._eye %s phase %s " % ( repr(view._eye), phase )) 
        self.z = z
        self.f = 0.
        self.phase = phase
        log.info("  initial._eye %s " % repr(self._eye) ) 

    sincosf = property(lambda self:tuple([
             math.sin(self.phase+(2.*math.pi*self.f)),
             math.cos(self.phase+(2.*math.pi*self.f))
             ]))

    # override attempt, to make rotations act about eye point : but not working as hoped
    #distance = property(lambda self:0)
    #translate_look2eye = property(lambda self:np.identity(4))
    #translate_eye2look = property(lambda self:np.identity(4))

    def _get__radius(self):
        return np.linalg.norm(self.basis_view._gaze[:2])
    _radius = property(_get__radius)   

    def _get__eye(self):
        s,c = self.sincosf
        return self._radius*np.array([c, s, 0]) + np.array([0,0,self.z])
    _eye = property(_get__eye)

    def _get__look_tangential(self):
        s,c = self.sincosf
        return self._radius*np.array([-s, c, 0])

    def _get__look(self):
        return self.basis_view._look
    _look = property(_get__look)

    def _get__up(self):
        #return self.basis_view._up
        return np.array([0,0,1])
    _up = property(_get__up)


    def _get_eye_look_up_model(self):
        """
        Parametrically determined 9 element array with model frame eye, look, up.
        """
        return np.vstack([np.append(self._eye,1),np.append(self._look,1),np.append(self._up,0)])
    eye_look_up_model = property(_get_eye_look_up_model)     

    def _get_eye_look_up_world(self):
        """
        Parametrically determined 3,4 shape array with world frame eye, look, up.
        """
        return self.current_view.model2world.matrix.dot( self.eye_look_up_model.T ).T 
    eye_look_up_world = property(_get_eye_look_up_world)     

    def _get_eye_look_up(self):
        """
        Parametrically determined 9, shape array with world frame eye, look, up.
        """
        elu = self.eye_look_up_world                                 # shape 3,4 array
        return np.concatenate( [elu[0,:3], elu[1,:3], elu[2,:3]])    # shape 9, array
    eye_look_up = property(_get_eye_look_up)     


    def __call__(self, fraction, bump=False):
        self.f = fraction


 
class DAEInterpolateView(DAEViewpointBase):
    """
    could swap `current_view` as f > 0.5, no too confusing 
    keep current as the "A" view, 
    only moving to the next view at the next cycle 
    (when the original "B" view becomes the "A" view of the next cycle)

    How to change animation speed without changing position 
    in the interpolation ?
    """
    interpolate = True
    def __init__(self, views):
        DAEViewpointBase.__init__(self) 
        assert len(views) > 1
        self.views = views
        self.cycle = 0
        self.define_pair(0,1)
        self.f = 0.

    nviews = property(lambda self:len(self.views))             # could add a view during operation ?
    current_view = property(lambda self:self.views[self.i])   
    next_view = property(lambda self:self.views[self.j])

    def smry(self):
        return "%s klop" % (self.__class__.__name__ )

    def _get_eye_look_up(self):
        """Linear interpolation of two 9 element arrays """
        return (1.-self.f)*self.a + self.f*self.b
    eye_look_up = property(_get_eye_look_up)     

    def smry(self):
        return " ".join([
                         "IV(%d->%d[%s] %d)" % (self.i,self.j,self.f,self.nviews), 
                         "A %s" % self.views[self.i],
                         "B %s" % self.views[self.j],
                        ])  

    def define_pair(self, i , j ):
        self.i = i  
        self.j = j
        log.debug("define_pair cycle %s i %s j %s : %s %s " % (self.cycle, i,j,  self.views[i], self.views[j])  )
        self.a = self.views[i].eye_look_up
        self.b = self.views[j].eye_look_up 
        self.cycle += 1

    def next_cycle(self):
        i = (self.i + 1) % self.nviews           
        j = (self.j + 1) % self.nviews            
        self.define_pair(i,j )

    def __call__(self, fraction, bump=False):
        """
        Consider sequence of view  A, B, C, D

        i,j = 0,1    a,b = A, B      as f approaches 1 are approaching B, bump sets the next pair 
              1,2          B, C      sawtooth plunges f to 0, so still at B and head off towards C 

        Getting flashbacks 
        """
        if bump:self.next_cycle()
        self.f = fraction         
            


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)


    nview = 2
    views = map(DummyView, np.arange(0,nview))
    print views

    view = DAEInterpolateView( views) 

    for frame in range(100):
        view(frame, 0.1)        
        #print view
        




    

