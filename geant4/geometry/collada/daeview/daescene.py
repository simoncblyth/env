#!/usr/bin/env python
"""
"""
import os, logging
log = logging.getLogger(__name__)
import numpy as np

from daetrackball import DAETrackball
from daecamera import DAECamera
from daeinterpolateview import DAEInterpolateView
from daeviewpoint import DAEViewpoint
from daeutil import Transform
from daelights import DAELights
from daetransform import DAETransform

# do not import anything that would initialize CUDA context here, for CUDA_PROFILE control from config
 

ivec_ = lambda _:map(int,_.split(","))
fvec_ = lambda _:map(float,_.split(","))

class DAEScene(object):
    """
    Keep this for handling state, **NOT interactivity**, **NOT graphics**     
    """
    def __init__(self, geometry, config ):
        self.geometry = geometry  

        self.config = config
        args = config.args
        self.set_toggles(args)
        self.speed = args.speed

        # nasty modal switch
        self.scaled_mode = args.scaled_mode  

        # trackball
        xyz = args.xyz if self.scaled_mode else (0,0,0)
        trackball = DAETrackball( thetaphi=config.thetaphi, xyz=xyz, trackballradius=args.trackballradius, translatefactor=args.translatefactor )
        self.trackball = trackball
  
        # view
        self.view = self.target_view( args.target , prior=None)
        if args.jump:
            self.view = self.interpolate_view(args.jump)

        # camera
        kscale = 1. if self.scaled_mode else config.args.kscale
        camera = DAECamera( size=config.size, kscale=kscale, near=args.near, far=args.far, yfov=args.yfov, nearclip=config.nearclip, farclip=config.farclip, yfovclip=config.yfovclip )
        self.camera = camera 

        # lights
        light_transform = Transform() if self.scaled_mode else geometry.mesh.model2world 
        self.lights = DAELights( light_transform, config )

        # kludge scaling  
        #self.kscale = kscale

        # Chroma raycaster, None if not --with-chroma
        self.raycaster = self.make_raycaster( config, geometry ) 

        # Image processor, None if not --with-cuda-image-processor
        self.processor = self.make_processor( config ) 

        # transform holds references to all relevant state-holders and is available to all
        transform = DAETransform( self ) 
        self.camera.transform = transform
        self.trackball.transform = transform
        self.view.transform = transform

        if not self.raycaster is None:
            self.raycaster.transform = transform

        self.transform = transform

        self.solids = []    # selected solids
        self.bookmarks = {} # bookmarked viewpoints

    def resize(self, size):
        self.camera.resize(size)
        if self.processor is not None:
            self.processor.resize(size)
        if self.raycaster is not None:
            self.raycaster.resize(size)

    def make_raycaster(self, config, geometry ):
        if not config.args.with_chroma:return None
        log.info("creating Chroma raycaster processor, CUDA_PROFILE %s " % os.environ.get('CUDA_PROFILE',"not-defined") )
        import pycuda.gl.autoinit
        from daeraycaster import DAERaycaster     
        raycaster = DAERaycaster( config, geometry )
        return raycaster
 
    def make_processor( self, config ):
        if not config.args.with_cuda_image_processor:return None
        size = config.size
        procname = config.args.cuda_image_processor
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
        self.cuda = args.cuda and args.with_cuda_image_processor
        self.animate = False
        self.markers = args.markers
        self.raycast = args.raycast and args.with_chroma 
        # 
        self.toggles = ("light","fill","line","transparent","parallel","drawsolid","markers")  # animate, raycast, cuda have separate handling

    def toggle(self, name):
        setattr( self, name , not getattr(self, name)) 

    def toggle_animate(self):
        if self.view.interpolate:
            self.toggle('animate')
        else:
            log.warn("cannot toggle animate as current view not interpolatable")

    def toggle_raycast(self):
        if self.config.args.with_chroma:
            self.toggle("raycast") 
        else:
            log.warn("cannot toggle --raycast unless launched --with-chroma")

    def toggle_cuda(self):
        if self.config.args.with_cuda_image_processor:
            self.toggle(k) 
        else:
            log.warn("cannot toggle --cuda unless launched --with-cuda-image-processor")

    def animation_speed(self, factor ):   
        self.speed *= factor

    def __repr__(self):
        return "SC " + str(self.transform) 

    def __str__(self):
        return " ".join(map(str,[self.transform, self.camera])) 

    def where(self):
        print str(self)
        eye = self.transform.eye[:3] 
        solids = self.containing_solids( eye )
        log.info("solids containing eye point %s " % repr(eye))   
        print "\n".join(map(repr,solids))

    def containing_solids(self, xyz ):
        """
        Find solids that contain the world frame coordinates argument,  
        sorted by extent.
        """
        indices = self.geometry.find_bbox_solid( xyz )
        solids = sorted([self.geometry.solids[_] for _ in indices],key=lambda _:_.extent) 
        return solids


    def pick_solid(self, click):
        solids = self.containing_solids( click )
        self.solids = solids
        if len(solids) == 0:
            log.warn("clicked_point %s found no containing solids : how did you manage that ?" % repr(click) )
            return None
        pass
        log.info("pick_solid selects %s solids smallest \n%s" % ( len(self.solids), solids[0] ))
        return solids[0]

    def clicked_point(self, click, target_mode ):
        """
        :param click: world frame xyz 

        In target mode this jumps to a new view of clicked solid, from a default viewpoint.  
        This is jarring as usually does not match the viewpoint from which the click was made.
        """ 
        solid = self.pick_solid(click)
        newview = None
        if target_mode:
            log.info("as target mode changing view to the new solid, index %s " % solid.index )
            newview = self.target_view( solid.index , prior=None )
            self.transform.equivalent_eye_look_up( solid )

        if newview is None:
            log.debug("view unchanged by clicked_point")
        else:
            log.info("view changed by clicked_point")
            self.view = newview

    def create_bookmark(self, click, numkey ):
        """
        :param click: world frame xyz 
        """ 
        solid = self.pick_solid(click)
        if solid is None:
            log.warn("create_bookmark: key %s failed as a solid was not clicked" % numkey )
            return
        log.info("create_bookmark: key %s solid %s" % (numkey,solid) )
        view = self.transform.spawn_view_jumping_frame(solid)
        self.bookmarks[numkey] = view
 
    def visit_bookmark(self, numkey):
        view = self.bookmarks.get(numkey,None)
        if view is None:
            log.warn("visit_bookmark: no such bookmark %s " % numkey)
            return
        self.update_view(view) 
        
  
    def external_message(self, msg ):
        """
        """ 
        live_args = self.config( msg )
        if live_args is None:
            log.warn("external_message [%s] PARSE ERROR : IGNORING " % str(msg)) 
            return
        pass
        log.info("external_message [%s] [%s]" % (msg,str(live_args))) 

        newview = None
        elu = {}
        raycast_config = {}
        for k,v in vars(live_args).items():
            if k == "target":
                newview = self.target_view(v, prior=self.view ) 
            elif k == "jump":
                newview = self.interpolate_view(v) 
            elif k == "ajump":
                newview = self.interpolate_view(v, append=True) 
            elif k == "raycast":
                self.toggle_raycast() 
            elif k == "cuda":
                self.toggle_cuda() 
            elif k in self.toggles:
                self.toggle(k)
            elif k in ("launch","block","flags",):
                raycast_config[k] = ivec_(v)
            elif k in ("max_time","alpha_depth","allsync",):
                raycast_config[k] = v
            elif k in ("eye","look","up"):
                elu[k] = v
            elif k in ("kscale","near","far","yfov","nearclip","farclip","yfovclip"):
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
            self.update_view(newview)

        if len(raycast_config)>0:
            if self.raycast:
                self.raycaster.reconfig(**raycast_config)
            else:
                log.warn("cannot reconfig raycaster without init option --with-chroma ")
            pass

        if len(elu) > 0:
            log.info("home-ing trackball and changing parameters of existing view %s " % repr(elu)) 
            self.trackball.home()
            self.view.current_view.change_eye_look_up( **elu )


    def update_view(self, newview ):
        self.trackball.home()
        self.view = newview

    def target_view(self, tspec, prior=None):
        log.info("target_view tspec[%s]" % tspec  )
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


