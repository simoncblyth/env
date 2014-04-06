#!/usr/bin/env python
"""
"""
import logging
log = logging.getLogger(__name__)
import numpy as np
from daetrackball import DAETrackball
from daecamera import DAECamera
from daeinterpolateview import DAEInterpolateView
from daeviewpoint import DAEViewpoint
from daeutil import Transform
from daelights import DAELights
from daeraycaster import DAERaycaster


ivec_ = lambda _:map(int,_.split(","))
fvec_ = lambda _:map(float,_.split(","))

class DAEScene(object):
    """
    Keep this for handling state, **NOT interactivity**, **NOT graphics**     
    """
    def __init__(self, geometry, config ):
        self.geometry = geometry  
        self.config = config
        pass
        args = config.args
        self.set_toggles(args)
        self.speed = args.speed
        self.scaled_mode = args.target is None  

        # CUDA processor
        self.processor = self.make_processor( config )

        # trackball
        xyz = args.xyz if self.scaled_mode else (0,0,0)
        trackball = DAETrackball( thetaphi=config.thetaphi, xyz=xyz, trackballradius=args.trackballradius, translatefactor=args.translatefactor )
        self.trackball = trackball
  
        # view
        self.view = self.change_view( args.target , prior=None)
        if args.jump:
            self.view = self.interpolate_view(args.jump)
        print self.view.smry()

        # camera
        camera = DAECamera( size=config.size, near=args.near, far=args.far, yfov=args.yfov, nearclip=config.nearclip, farclip=config.farclip, yfovclip=config.yfovclip )
        self.camera = camera 

        # lights
        light_transform = Transform() if self.scaled_mode else geometry.mesh.model2world 
        self.lights = DAELights( light_transform, config )

        # kludge scaling  
        kscale = 1. if self.scaled_mode else config.args.kscale
        self.kscale = kscale

        # raycaster
        self.raycaster = DAERaycaster( config, geometry )

        # selected solids
        self.solids = []


    def resize(self, size):
        self.camera.resize(size)
        if self.processor is not None:
            self.processor.resize(size)

    def _get_pixel2world(self):
        """ 
        Provides pixel2world matrix that transforms pixel coordinates like (0,0,0,1) or (1023,767,0,1)
        into corresponding world space locations at the near plane for the current camera and view. 

        Unclear where best to implement this : needs camera, view  and kscale
        """
        scale = np.identity(4)   # it will be getting scaled down so have to scale it up, annoyingly 
        scale[0,0] = self.kscale
        scale[1,1] = self.kscale
        scale[2,2] = self.kscale

        pixel2camera_scaled = np.dot( scale, self.camera.pixel2camera ) # order matters, have to scale pixel2camera, not after camera2world
        camera2world = self.view.camera2world.matrix
        pixel2world = np.dot( camera2world, pixel2camera_scaled )   
        return pixel2world
    pixel2world = property(_get_pixel2world)

 
    def make_processor( self, config ):
        if not config.args.with_cuda:return None
        size = config.size
        procname = config.args.processor
        log.info("creating CUDA processor : %s " % procname )
        import pycuda.gl.autoinit
        from env.pycuda.pycuda_pyopengl_interop import Invert, Generate
        if procname == "Invert":
            processor = Invert(size)
        elif procname == "Generate":
            processor = Generate(size)
        else:
            processor = None
            log.warn("failed to create CUDA processor %s " % procname )
        return processor
 
    def set_toggles(self, args):
        self.light = args.light
        self.fill = args.fill
        self.line = args.line
        self.transparent = args.transparent
        self.parallel = args.parallel
        self.drawsolid = False
        self.cuda = args.cuda
        self.animate = False
        self.markers = args.markers
        self.raycast = args.raycast
        # 
        self.toggles = ("light","fill","line","transparent","parallel","drawsolid","cuda","animate","markers","raycast")

    def toggle_attr(self, name):
        setattr( self, name , not getattr(self, name)) 

    def animation_speed(self, factor ):   
        self.speed *= factor

    def where(self):
        model_xyz = self.view.offset_eye_position( self.trackball.xyz ) 
        return model_xyz

    def __repr__(self):
        w = self.where()
        return "SC %5.1f %5.1f %5.1f " % (w[0],w[1],w[2])

    def clicked_point(self, click ):
        """
        :param click: world frame xyz 

        Find solids that contain the click coordinates,  
        sorted by extent.
        """ 
        indices = self.geometry.find_bbox_solid( click )
        solids = sorted([self.geometry.solids[_] for _ in indices],key=lambda _:_.extent) 
        self.solids = solids

    def bookmark(self):
        log.info("bookmark") 
        print self.view.current_view

    def external_message(self, msg ):
        live_args = self.config( msg )
        if live_args is None:
            log.warn("external_message [%s] PARSE ERROR : IGNORING " % str(msg)) 
            return
        pass
        log.info("external_message [%s] [%s]" % (msg,str(live_args))) 

        newview = None
        elu = {}
        for k,v in vars(live_args).items():
            if k == "target":
                newview = self.change_view(v, prior=self.view ) 
            elif k == "jump":
                newview = self.interpolate_view(v) 
            elif k == "ajump":
                newview = self.interpolate_view(v, append=True) 
            elif k in self.toggles:
                self.toggle_attr(k)
            elif k in ("eye","look","up") :
                elu[k] = v
            elif k == "kscale":
                self.kscale = kscale
            elif k in ("near","far","yfov","nearclip","farclip","yfovclip"):
                setattr(self.camera, k, v )
            elif k in ("translatefactor","trackballradius"):
                setattr(self.trackball, k, v )
            else:
                log.info("handling of external message key [%s] value [%s] not yet implemented " % (k,v) )
            pass
        pass

        if newview is None:
            log.debug("view unchanged by external message")
        else:
            log.info("view changed by external message")
            self.view = newview

        if len(elu) > 0:
            log.info("changing parameters of existing view %s " % repr(elu)) 
            self.view.current_view.change_eye_look_up( **elu )
 

    def change_view(self, tspec, prior=None):
        log.info("change_view tspec[%s]" % tspec  )
        self.trackball.home()
        return DAEViewpoint.make_view( self.geometry, tspec, self.config.args, prior=prior )
     
    def interpolate_view(self, jspec, append=False):
        self.trackball.home()
        views  = self.view.views if append else [self.view.current_view]
        views += [DAEViewpoint.make_view( self.geometry, j, self.config.args, prior=views[-1] ) for j in jspec.split(":")]
        log.info("interpolated_view append %s movie sequence with %s views " % (append,len(views)))

        interpolateview = None
        if len(filter(None, views)) != len(views):
            log.warn("interpolate_view encountered None view : geometry nodes %s not loaded ? " % jspec ) 
        else:
            interpolateview = DAEInterpolateView(views)
        pass
        return interpolateview

    def dump(self):
        print "view\n", self.view
        print "trackball\n", self.trackball


if __name__ == '__main__':
    pass


