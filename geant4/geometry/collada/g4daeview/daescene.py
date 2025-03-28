#!/usr/bin/env python
"""
"""
import os, sys, logging
log = logging.getLogger(__name__)
import numpy as np
import glumpy as gp
from glumpy.window import event as window_event

from daetrackball import DAETrackball
from daeinterpolateview import DAEInterpolateView
from daeviewpoint import DAEViewpoint
from daeutil import Transform
from daelights import DAELights
from daetransform import DAETransform
from daebookmarks import DAEBookmarks
from daeanimator import DAEAnimator
from daeevent import DAEEvent



# do not import anything that would initialize CUDA context here, for CUDA_PROFILE control from config
 

ivec_ = lambda _:map(int,_.split(","))
fvec_ = lambda _:map(float,_.split(","))


class DAEChromaContextDummy(object):
    raycaster = None
    propagator = None
    dummy = True

class DAEScene(window_event.EventDispatcher):
    """
    Keep this for handling state, **NOT interactivity**, **NOT graphics**     
    """
    def __init__(self, geometry, chroma_geometry, config ):
        """
        :param geometry: DAEGeometry instance
        :param chroma_geometry: chroma.Detector or chroma.Geometry instance
        :param config: DAEConfig instance
        """
        self.geometry = geometry  
        self.config = config

        if self.config.args.with_chroma:
            from daechromacontext import DAEChromaContext     
            self.chroma = DAEChromaContext( config, chroma_geometry, gl=1 )
        else:
            self.chroma = DAEChromaContextDummy()
        pass

        args = config.args
        self.set_toggles(args)

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

        # lights
        light_transform = Transform() if self.scaled_mode else geometry.mesh.model2world 
        self.lights = DAELights( light_transform, config )

        # bookmarked viewpoints, now contains DAECamera and DAEClipper instances
        self.bookmarks = DAEBookmarks(config, geometry ) 

        # Chroma raycaster and propagator, None if -C/--nochroma
        self.raycaster = self.chroma.raycaster
        self.propagator = self.chroma.propagator 

        # Image processor, None if not --with-cuda-image-processor
        self.processor = self.make_processor( config ) 

        # transform holds references to all relevant state-holders 
        transform = DAETransform( self ) 

        # make transfrom available to all, it represents the trackballing actions of the user 
        self.transform = transform
        self.camera.transform = transform
        self.trackball.transform = transform
        self.view.transform = transform
        self.bookmarks.transform = transform

        if not self.raycaster is None:
            self.raycaster.transform = transform

        # Event handling, either root file load/save or network messages 
        # needs to be after transform setup as may do a launch load which adds auto-bookmark 9 
        log.info("**********  scene.event creation ")
        self.event = DAEEvent(config, self) 


        self.progpoint = True

        self.solids = []    # selected solids

        # highlight volumes via subvbo
        self.touch_index = None
        self.touch_mesh = None

        # bookmark 0 : corresponding to launch viewpoint 
        self.bookmarks.create_for_solid(self.view.solid, 0)

        # animation frame count
        self.animator = DAEAnimator(args.period)
        log.info("**********  scene creation DONE ")

    def external_npy(self, npy ):
        log.info("external_npy NPY received")
        self.event.external_npy( npy )


    clipper = property(lambda self:self.bookmarks.clipper)
    camera = property(lambda self:self.bookmarks.camera)


    def reset_count(self):
        self.animator.reset()

    def animation_period(self, factor ):   
        self.animator.change_period(factor)

    def tick(self, dt):
        #log.info("tick") 
        fraction, bump = self.animator() 
        self.view(fraction, bump)

    def resize(self, size):
        self.camera.resize(size)
        if self.processor is not None:
            self.processor.resize(size)
        if self.raycaster is not None:
            self.raycaster.resize(size)

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
        self.drawsolid = False
        self.photonmagic = True
        self.cuda = args.cuda and args.with_cuda_image_processor
        self.animate = False
        self.markers = args.markers
        self.raycast = args.raycast and args.with_chroma 
        self.showmetric = False
        # 
        self.toggles = ("light","fill","line","transparent","drawsolid","markers",)  # animate, raycast, cuda have separate handling

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

    def toggle_showmetric(self):
        if self.config.args.with_chroma:
            self.toggle("showmetric") 
            self.raycaster_reconfig( showmetric=self.showmetric )
        else:
            log.warn("cannot toggle --showmetric unless launched --with-chroma")

    def __repr__(self):
        return "SC " + str(self.transform) 

    def __str__(self):
        return " ".join(map(str,[self.geometry, self.transform, self.camera])) 

    def exit(self):
        self.bookmarks.save()
        if not self.raycaster is None:
            self.raycaster.exit() 
        print "\n%s %s\n" % (os.path.basename(sys.argv[0]).replace(".py",".sh"), str(self))

    here = property(lambda self:self.transform.eye[:3])

    def where(self):
        print str(self)
        here = self.here
        solids = self.geometry.containing_solids( here )

        self.bookmarks.add_clipping_plane( self.transform.plane )

        log.info("solids containing eye point %s " % repr(here))   
        print "\n".join(map(repr,solids))


    def pick_solid(self, click):
        solids = self.geometry.containing_solids( click )
        self.solids = solids
        if len(solids) == 0:
            #log.warn("clicked_point %s found no containing solids : how did you manage that ?" % repr(click) )
            return None
        pass
        log.debug("pick_solid selects %s solids smallest \n%s" % ( len(self.solids), solids[0] ))
        return solids[0]

    def clicked_point(self, click, target_mode ):
        """
        :param click: world frame xyz 

        In target mode this jumps to a new view of clicked solid, from a default viewpoint.  
        This is jarring as usually does not match the viewpoint from which the click was made.
        """ 
        log.debug("clicked_point click %s  target_mode %s " % (repr(click), target_mode)) 
        self.event.clicked_point( click )

        solid = self.pick_solid(click)
        if solid:
            log.debug("touched solid.index %s solid.solidindex %s " % (solid.index,solid.solidindex))
            self.update_touch_mesh(solid.solidindex) 
        else:
            log.debug("null solid")

        view = None
        if target_mode:
            log.debug("as target mode changing view to the new solid, index %s " % solid.index )
            view = self.target_view( solid.index , prior=None )
        pass
        if view is None:
            log.debug("view unchanged by clicked_point")
        else:
            log.debug("view changed by clicked_point")
            self.update_view(view)

    def update_touch_mesh(self, index, enabled=True):
        """
        The touch_mesh is drawn on top of the full one in order
        to "highlight" touched volumes
        """
        if index == self.touch_index or index == -1:pass 
        log.debug("update_touch_mesh index %s " % index )
        if enabled:
            touch_vbo = self.geometry.make_vbo(scale=self.scaled_mode, rgba=self.config.rgba, index=index)
            self.touch_mesh = gp.graphics.VertexBuffer( touch_vbo.data, touch_vbo.faces )
        pass

    def create_bookmark(self, click, numkey):
        """
        :param click: world frame xyz 
        """ 
        solid = self.pick_solid(click)
        if solid is None:
            log.warn("create_bookmark: key %s failed as a solid was not clicked" % numkey )
            return
        pass
        self.bookmarks.create_for_solid( solid, numkey )
 
    def visit_bookmark(self, numkey ):
        view = self.bookmarks.visit(numkey)
        if view is None:
            log.warn("visit_bookmark: no such bookmark %s " % numkey)
            return 
        else:
            log.debug("visit_bookmark %s " % numkey )
        pass
        self.update_view(view) 

    def update_current_bookmark(self):
        log.info("update_current_bookmark")
        self.bookmarks.update_current()

    def setup_bookmark_interpolation(self):
        """
        Maybe should start from current bookmark ?
        """ 
        log.info("setup bookmark interpolation")
        view = self.bookmarks.make_interpolate_view()
        if view is None:
            log.warn("failed to make_interpolate_view")
        else:
            self.update_view(view)

    def setup_parametric_interpolation(self):
        """
        Orbiting/Flyaround mode
        """
        log.info("setup parametric interpolation")
        view = self.bookmarks.make_parametric_view()
        self.update_view(view)

    def save_to_file(self):
        """
        Invoke with `udp.py --screenshot`
        """
        if not hasattr(self, 'fig_handler'):
            log.warn("no scene.fig_handler cannot save_to_file from external_message ")
            return
        pass
        log.info("save_to_file")
        self.fig_handler.save_to_file()

    def external_message(self, msg ):
        """
        TODO: 

        #. pull this out into a separate class
        #. distribute the properties handled lists into the classes where they are handled

        """ 
        live_args = self.config.live_parse( msg )
        if live_args is None:
            log.warn("external_message [%s] PARSE ERROR : IGNORING " % str(msg)) 
            return
        pass
        log.info("external_message [%s] [%s]" % (msg,str(live_args))) 

        newview = None
        elu = {}
        raycast_config = {}
        event_config = []
        photon_config = []

        for k,v in vars(live_args).items():
            if k == "target":
                newview = self.target_view(v, prior=self.view ) 
            elif k == "object":
                newview = self.object_view(v, prior=self.view ) 
            elif k == "jump":
                newview = self.interpolate_view(v) 
            elif k == "ajump":
                newview = self.interpolate_view(v, append=True) 
            elif k == "raycast":
                self.toggle_raycast() 
            elif k == "cuda":
                self.toggle_cuda() 
            elif k == "screenshot":
                self.save_to_file() 
            elif k in self.toggles:
                self.toggle(k)
            elif k in ("launch","block","flags",):
                raycast_config[k] = ivec_(v)
            elif k in ("max_time","alpha_depth","allsync",):
                raycast_config[k] = v
            elif k == "showmetric":
                raycast_config[k] = v
                self.toggle_showmetric() 
            elif k in ("save","load","key","reload","clear","type","slice"):
                event_config.append( (k,v,) )   
            elif k in ("fpholine","fphopoint","tcut","mask","bits","time", "style","pid","mode","timerange","cohort","material","sid","surface",):
                photon_config.append( (k,v,) )   
            elif k in ("eye","look","up"):
                elu[k] = v
            elif k in self.camera.reconfigurables:
                setattr(self.camera, k, v )
            elif k in ("translatefactor","trackballradius"):
                setattr(self.trackball, k, v )
            elif k in ("propagate",):
                log.info("setting config.args.%s = %s " % (k,v ))
                setattr(self.config.args,k,v)  # problem with this is no immediate action, it just takes effect on next load
            else:
                log.info("handling of external message key [%s] value [%s] not yet implemented " % (k,v) )
            pass
        pass

        if newview is None:
            log.debug("view unchanged by external message")
        else:
            log.info("view changed by external message")
            self.update_view(newview)

        if len(event_config) > 0:
            self.event.reconfig(event_config)

        if len(photon_config) > 0:
            log.info("photon_config %s " % repr(photon_config))
            self.event.dphotons.reconfig(photon_config)

        if len(raycast_config)>0:
            self.raycaster_reconfig(**raycast_config)

        if len(elu) > 0:
            log.info("home-ing trackball and changing parameters of existing view %s " % repr(elu)) 
            self.trackball.home()
            self.view.current_view.change_eye_look_up( **elu )



    def raycaster_reconfig(self, **raycast_config ):
        if not self.raycast:
            log.warn("cannot reconfig raycaster without init option --with-chroma ")
            return
        self.raycaster.reconfig(**raycast_config)

    def update_view(self, newview ):
        """
        Replaces scene.view instance with another view instance, which 
        can be of any view type:

        #. DAEViewpoint
        #. DAEInterpolatedView
        #. DAEParametricView

        """
        if newview is None:
            log.warn("update_view received newview None")
            return 

        self.trackball.home()
        if newview.interpolate:
            self.reset_count()
        else:
            self.animate = False
        pass
        self.view = newview

    def target_view(self, tspec, prior=None):
        log.debug("target_view tspec[%s]" % tspec  )
        return DAEViewpoint.make_view( self.geometry, tspec, self.config.args, prior=prior )

    def object_view(self, ospec="0", prior=None):
        log.debug("object_view ospec[%s]" % ospec  )
        return DAEViewpoint.make_object_view( self.event, ospec, self.config.args, prior=prior )

    def loadnext(self):
        self.event.loadnext()

    def reload_(self):
        self.event.reload_()

    def loadprev(self):
        self.event.loadprev()

    def step(self):
        self.event.step(self.chroma)

    def interpolate_view(self, jspec, append=False):
        """
        TODO: Maybe get rid of this, interpolation based on bookmarks is much better 
        """
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

    def dispatch(self,event_name='on_needs_redraw',event_obj=None):
        log.info("dispatch %s %s " % (event_name, event_obj))
        self.dispatch_event(event_name, event_obj)
        
    def dump(self):
        print "view\n", self.view
        print "trackball\n", self.trackball

DAEScene.register_event_type('on_needs_redraw')


if __name__ == '__main__':
    pass


