#!/usr/bin/env python

import logging, math
log = logging.getLogger(__name__)
import numpy as np
from daeutil import printoptions, WorldToCamera, CameraToWorld, Transform, translate_matrix, scale_matrix


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

    def pixel2world_matrix(self, camera ):
        """ 
        Provides pixel2world matrix that transforms pixel coordinates like (0,0,0,1) or (1023,767,0,1)
        into corresponding world space locations at the near plane for the current camera and view. 

        Unclear where best to implement this : needs camera, view  and kscale

        TODO: accomodate trackball offsets, so can raycast without being homed on a view
        """
        iscale = scale_matrix( camera.kscale )        # it will be getting scaled down so have to scale it up, annoyingly 
        return reduce(np.dot, [ self.camera2world.matrix, iscale, camera.pixel2camera ])


    def modelview_matrix(self, trackball, kscale ):
        """
        Objects are transformed from **world** space to **eye** space using GL_MODELVIEW matrix, 
        as daeviewgl regards model spaces as just input parameter conveniences
        that OpenGL never gets to know about those.  

        So need to invert MODELVIEW and apply it to the origin (eye position in eye space)
        to get world position of eye.  Can then convert that into model position.  

        Motivation:

           * determine effective view point (eye,look,up) after trackballing around

        The MODELVIEW sequence of transformations in daeframehandler in OpenGL reverse order, 
        defines exactly what the trackball output means::

            kscale = self.camera.kscale
            distance = view.distance
            gl.glScalef(1./kscale, 1./kscale, 1./kscale)   
            gl.glTranslate ( *trackball.xyz )       # former adhoc 1000. now done internally in trackball.translatefactor
            gl.glTranslate ( 0, 0, -distance )      # shunt back, eye back to origin                   
            gl.glMultMatrixf (trackball._matrix )   # rotation around "look" point
            gl.glTranslate ( 0, 0, +distance )      # look is at (0,0,-distance) in eye frame, so here we shunt to the look
            glu.gluLookAt( *view.eye_look_up )      # NB no scaling, still world distances, eye at origin and point -Z at look

        To get the unproject to dump OpenGL modelview matrix, touch a pixel.

        * http://stackoverflow.com/questions/4964101/pep-3118-warning-when-using-ctypes-array-as-numpy-array


        Using nomenclature

        #. **eye frame** is trackballed and **down scaled** and corresponds to GL_MODELVIEW 
        #. **camera frame** is just from eye, look, up

        #. this means eye frame distance between eye and look needs to be down scaled

        """
        distance = self.distance       # eye frame translation along -Z

        world2camera = self.world2camera.matrix

        to_look   = translate_matrix((0,0, distance)) 

        # RuntimeWarning: Item size computed from the PEP 3118 buffer format string does not match the actual item size
        trackball_rot = np.array( trackball._matrix, dtype=float).reshape(4,4).T   # transposing to match GL_MODELVIEW
        from_look = translate_matrix((0,0,  -distance)) 
        trackball_tra = translate_matrix(trackball.xyz)   # trackball.xyz 3-element np array

        down_scale = scale_matrix( 1./kscale )

        transforms = [down_scale, trackball_tra, from_look, trackball_rot, to_look, world2camera ]
        world2eye = reduce(np.dot, transforms)



        up_scale = scale_matrix( kscale )
        camera2world = self.camera2world.matrix
        trackball_itra = translate_matrix(-trackball.xyz) 
 
        itransforms = [  camera2world, from_look, trackball_rot.T, to_look, trackball_itra, up_scale ]
        eye2world = reduce(np.dot, itransforms )

        #check = np.dot( eye2world, world2eye )
        #assert np.allclose( check, np.identity(4) ), check   # close, but not close enough in translate column

        #if 0:
        #    print "world2eye\n%s " % world2eye 
        #    print "eye2world\n%s " % eye2world 
        #    print "check \n%s" % check

        return world2eye, eye2world


    def offset_eye_look_up(self, trackball, kscale ):
        """
        :param trackball: DAETrackball instance
        :return: model frame coordinates of offset eye, look, up  position

        Original eye of the view is semi-fixed.
        Trackball translations do not change the view instance eye.

        Prior approaches to trackball handling that caused confusion
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        #. treating trackball.xyz as an offset
        #. treating trackball.xyz as an absolute position to be transformed
     
        Testing using only the simple special case of translations in the gaze line
        (Z panning backwards) was highly misleading as several incorrect treatments 
        worked for this case but not in general.

        Successful treatment:

        #. consider trackball as a source of translation and rotation transforms
           NOT as providing a coordinate to be transformed 
        #. work with entire MODELVIEW transform sequence at once rather than 
           attempting to operate with partial sequences

        Testing trackball pan conversion to model position
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 
        Standard position to debug from with a wide view 
        and remote command to move view around numerically::

            daeviewgl.py -t 8005 --with-chroma --cuda-profile --near 0.5 --size 640,480
            udp.py --eye=10,10,10
            udp.py --eye=15.5,-6.5,30.2   
            # remote commands change the base view, 
            # so must home the trackball for correspondence with what you see
                 
        #. use remote command to set position `udp.py --eye=10,0,0` (this will home the trackball and change the view)
        #. check the scene "where" position in title bar (after "SC") and eye position (after "e") are the same   
        #. use trackball pan controls (eg spacebar drag down) to move in +Z_eye direction, the SC position should 
           update while the base e position stays fixed, for example ending at SC 20,0,0 and e 10,0,0
        #. issue another remote command, which homes the trackball and sets the view to the 
           SC position `udp.py --eye=20,0,0` there should be no visual jump and the
           base view position  `e 20,0,0` should now match, as have homed


        Remaining mysteries
        ~~~~~~~~~~~~~~~~~~~~ 

        #. Using (0,0,-d,1) with `d=self.distance` for the look point 
           leads to crazy look positions after trackballing around, 
           whereas just using (0,0,-1,1) doesnt


        Confusion between frames. distance is expressed in world frame units, it
        needs some scaling to be usable in model frame.

        """
        world2eye, eye2world = self.modelview_matrix( trackball, kscale ) 

        log.info("world2eye \n%s" % str(world2eye))
        log.info("eye2world \n%s" % str(eye2world))

        #eye2world_1 = np.linalg.inv(world2eye)
        #assert np.allclose( eye2world_1, eye2world )

        eye2model = np.dot( self.world2model.matrix, eye2world ) 

        eye_distance = self.distance / kscale
        # canonical eye/look/up in the eye frame, which stays valid by definition
        eye_look_up_eye = np.vstack([[0,0,0,1],[0,0,-eye_distance,1],[0,eye_distance,0,0]]).T  # eye frame
        log.info("eye_look_up_eye \n%s" % str(eye_look_up_eye))

        eye_look_up_model = eye2model.dot(eye_look_up_eye)                                        # model frame

        log.info("eye_look_up_model \n%s" % str(eye_look_up_model))

        return np.split( eye_look_up_model.T.flatten(), 3 )


    def offset_where(self, trackball, kscale ):

        eye, look, up = self.offset_eye_look_up( trackball, kscale ) 

        s_ = lambda name:"--%(name)s=%(fmt)s" % dict(fmt="%s",name=name) 
        fff_ = lambda name:"--%(name)s=\"%(fmt)s,%(fmt)s,%(fmt)s\"" % dict(fmt="%5.1f",name=name) 

        return   " ".join(map(lambda _:_.replace(" ",""),[
                         s_("target") % self.target,
                         fff_("eye")  % tuple(eye[:3]), 
                         fff_("look") % tuple(look[:3]), 
                         fff_("up")   % tuple(up[:3]),
                         fff_("norm") % tuple([np.linalg.norm(eye[:3]), np.linalg.norm(look[:3]), np.linalg.norm(up[:3])]),
                          ])) 



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

    def _get_eye_look_up_model(self):
        return np.vstack([np.append(self._eye,1),np.append(self._look,1),np.append(self._up,0)])
    eye_look_up_model = property(_get_eye_look_up_model, doc="3x4 element array containing eye, look, up in homogenous model frame coordinates" )

    def _get_eye_look_up_world(self):
        return self.model2world.matrix.dot( self.eye_look_up_model.T ).T 
    eye_look_up_world = property(_get_eye_look_up_world, doc="3x4 element array containing eye, look, up in homogenous world frame coordinates" )


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



def test_0():

    pixel2camera = camera.pixel2camera
    camera2world = view.camera2world.matrix
    pixel2world = np.dot( camera2world, pixel2camera )

    corners = np.array(camera.pixel_corners.values())

    worlds  = np.dot( corners, pixel2world.T )
    worlds2 = np.dot( pixel2world, corners.T ).T   
    assert np.allclose( worlds, worlds2 )



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





 
