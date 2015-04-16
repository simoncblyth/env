#!/usr/bin/env python
"""

"""
import os, sys, logging
from datetime import datetime

try:
    from collections import OrderedDict 
except ImportError:
    OrderedDict = dict

log = logging.getLogger(__name__)
from glumpy.window import key
number_keys = (key._0,key._1,key._2,key._3,key._4,key._5,key._6,key._7,key._8,key._9,)

from daedispatcher import DAEDispatcher
try:
    from daeresponder import DAEResponder
except ImportError:
    DAEResponder = None

from daeviewport import DAEViewport
from daemenu import DAEMenu, DAEMenuDemo


t0, frames, t = 0,0,0


class DAEKeys(object):
    def __init__(self):
        key = OrderedDict()

        key["ESCAPE"] = "exit "

        key["*"] = "--- toggles ---"
        key["S"] = "toggle fullscreen "
        key["I"] = "toggle fill triangles"
        key["L"] = "toggle line"
        key["T"] = "toggle transparent"
        key["P"] = "toggle parallel projection (aka orthographic)"
        key["K"] = "toggle drawing illustration markers for: view frustum, lights, raycast rays, eye and look positions "
        key["O"] = "toggle drawing wireframe spheres to mark mouse/trackpad picked solids  "
        key["M"] = "toggle animation: when not in time mode toggles viewpoint interpolation (setup first with B or V), in time mode toggles propagation time scanning  "
        key["C"] = "toggle CUDA image processor, requires launch --with-cuda-image-processor "
        key["R"] = "toggle Chroma raycasting for the current view, requires launch --with-chroma "
        key["J"] = "toggle Chroma raycasting metric display eg pixel processing time, requires launch --with-chroma, reconfig bitshifts with eg --flags 15,0 "
        
        key["**"] = "--- viewing ---"
        key[""] = "default action without any key pressed, mouse/trackpad movement rotates around the *look* position" 
        key["Z"] = "z-translate, while pressed mouse/trackpad movement translates in screen Z in/out direction, modifiers SHIFT and/or OPTION speed up movement " 
        key["2finger"] = "2-finger click trackpad and drag, a non-modal alternative for Z translation"
        key["X"] = "xy-translate, while pressed mouse/trackpad movement translates x,y (left-right,up-down), modifiers SHIFT and/or OPTION speed up movement "
        key["N"] = "near, while pressed, mouse/trackpad movement changes near clipping plane "
        key["F"] = "far,while pressed, mouse/trackpad movement changes far clipping plane "
        key["Y"] = "yfov, while pressed, mouse/trackpad movement changes field of view angle "
        key["Q"] = "target mode, while pressed, picking a solid changes target to the solid "

        key["***"] = "--- factors ---"
        key["UP"] = "increase mouse/trackpad drag factor *2"
        key["DOWN"] = "decrease mouse/trackpad drag factor *0.5"
        key["RIGHT"] = "increase animation speed *2"
        key["LEFT"] = "decrease animation speed *0.5"
        key["]"] = "load next event"
        key["["] = "load previous event"


        key["****"] = "--- misc ---"
        key["H"] = "home, resets trackball translation and rotation offsets to zero "
        key["1-9"] = "create/visit bookmark: while pressing number key click solid to create bookmark, subsequently press number key to visit bookmark" 
        key["0"] = "bookmark zero is added at startup for the initial viewpoint/solid "
        key["SPACE"] = "update current bookmark: the last visited/created bookmark is updated to accomodate changed viewpoint from trackballing around "
        key["W"] = "where: write to stdout the commandline to recreate the current state of view and camera and make current near clipping plane into a fixed clipping plane "
        key["@"] = "shift-2: clear all fixed clipping planes "
        key["A"] = "timecut, while presses mouse/trackpad movement scans a globaltime cut applied to any loaded event objects"
        key["D"] = "single step chroma optical photon propagation, moving photon positions and directions to the next step of their simulation "
        key["U"] = "usage, write help text to stdout "
        key["B"] = "setup bookmark interpolation view, to toggle animation use M  "
        key["V"] = "setup parametric orbiting view based on current bookmark, to toggle animation use M  "
        key["E"] = "screen capture and write to timestamp dated .png file in current directory"
        key["G"] = "light: NOT YET IMPLEMENTED "

        self.key = key

    def __str__(self):
        return "\n".join(["%-10s : %s " % (k,v) for k,v in self.key.items()])   


class DAEInteractivityHandler(object):
    """
    Keep this for handling interactivity, **NOT** graphics, **NOT state**     
    """
    def __init__(self, fig, frame_handler, scene, config ):
        #
        self.fig = fig
        self.frame_handler = frame_handler
        self.scene = scene
        self.config = config
        pass
        self.dragfactor = config.args.dragfactor
        self.modfactor = 1
        self.fps = 0
        self.viewport = DAEViewport(map(int,config.args.size.split(",")))
        self.keys = DAEKeys()
        #
        self.scan_mode = False
        self.time_mode = False
        self.zoom_mode = False
        self.pan_mode = False
        self.near_mode = False
        self.far_mode = False
        self.yfov_mode = False
        self.target_mode = False
        self.bookmark_mode = False
        self.bookmark_key = None
        self.tab_mode = False
        #
        if config.args.fullscreen:
            self.toggle_fullscreen()
        pass

        fig.push(self)                     # event notification from fig 
        self.hookup_udp_dispatcher(config)     # event notification from dispatcher
        self.hookup_zmq_responder()

    def __repr__(self):
        return "H %5.2f %5.2f " % ( self.dragfactor, self.fps )

    def hookup_udp_dispatcher(self, config):
        """
        dispatcher collect messages over UDP, allowing remote control

        #. the `dispatcher.push_handlers(self)` identifies this 
           DAEInteractivityHandler instance as a handler of events 
           from the DAEDispatcher instance, ie the on_external_message event 

        """
        dispatcher = DAEDispatcher(port=config.args.port, host=config.args.host)
        log.debug(dispatcher)        

        def _check_dispatcher(dt):
            dispatcher.update()
        timer = self.fig.timer(5.)  # fps
        timer(_check_dispatcher) 

        dispatcher.push_handlers(self)   


    def hookup_zmq_responder(self):
        """
        TODO: workout how to switch off timers, see *glumpy-* on timers
        """
        if DAEResponder is None:
            log.warn("cannot use DAEResponder without ZMQ/pyZMQ " )
            return 

        zmq_responder = DAEResponder(self.config, self.scene)
        log.debug(zmq_responder)        

        def _check_zmq_responder(dt):
            zmq_responder.update()

        zmq_timer = self.fig.timer(5.)  # fps
        zmq_timer(_check_zmq_responder) 

        self.zmq_timer = zmq_timer

        zmq_responder.push_handlers(self)      # get event notification from responder


    def _get_title(self):
        rdr = self.scene.raycaster.renderer if self.scene.raycast else None
        return " ".join(map(repr,filter(None,[
                     rdr,
                     self.scene.event,
                     self.scene.bookmarks,
                     self.scene.transform,
                     self.scene.view,
                     self.scene.camera,
             #       self.frame_handler,
             #       self.scene.trackball,
                     self,
                     ])))
    title = property(_get_title)     

    def redraw(self):
        self.fig.window.set_title(self.title)
        self.fig.redraw()

    def retitle(self):
        self.fig.window.set_title(self.title)
 
    def toggle_fullscreen(self):
        if self.fig.window.get_fullscreen():
            self.fig.window.set_fullscreen(0)
        else:
            self.fig.window.set_fullscreen(1)
 
    def on_init(self):
        self.fig.window.set_title(self.title)

    def on_external_message(self, msg ):
        self.scene.external_message(msg)
        self.redraw()

    def on_external_npy(self, npy ):
        self.scene.external_npy(npy)
        self.redraw()
        return True   # prevent other handlers

    def on_needs_redraw(self, msg ):
        #log.info("on_needs_redraw")
        self.redraw()

    def on_resize(self, width, height, x=0, y=0):
        self.viewport.resize((width, height))
        self.scene.resize((width, height))

    def on_draw(self):
        self.fig.clear(0.85,0.85,0.85,1)  # seems to have no effect even when lighting disabled

    def exit(self):
        self.scene.exit()
        sys.exit() 

    def save_to_file(self):
        x,y,w,h = self.fig.viewport
        name = datetime.now().strftime("%Y%m%d-%H%M%S")
        log.info("save_to_file name %s" % name)
        self.frame_handler.write_to_file( name, x,y,w,h, outdir=self.config.args.outdir )

    def usage(self):
        print str(self.keys)



    dragmode = property(lambda self:self.zoom_mode or self.pan_mode or self.near_mode or self.far_mode or self.yfov_mode) 


    def on_key_press(self, symbol, modifiers):
        """
        ABCDEFGHIJKLMNOPQRSTUVWXYZ
        **************************
        """
        if   symbol == key.ESCAPE: self.exit()
        elif symbol == key.A: self.scan_mode = True            # used for qcut scan, was used for primitive time cutting before adding time_mode 
        elif symbol == key.QUOTELEFT: self.time_mode = True
        elif symbol == key.Z: self.zoom_mode = True
        elif symbol == key.X: self.pan_mode = True
        elif symbol == key.N: self.near_mode = True
        elif symbol == key.F: self.far_mode = True
        elif symbol == key.Y: self.yfov_mode = True
        elif symbol == key.Q: self.target_mode = True
        elif symbol == key.UP: self.dragfactor *= 2.
        elif symbol == key.DOWN: self.dragfactor *= 0.5
        elif symbol == key.S: self.toggle_fullscreen()
        elif symbol == key.LEFT: self.animation_period(2.0)
        elif symbol == key.RIGHT: self.animation_period(0.5)
        elif symbol == key.H: self.scene.trackball.home()
        elif symbol == key.SPACE: self.scene.update_current_bookmark()
        elif symbol == key.B: self.scene.setup_bookmark_interpolation()
        elif symbol == key.V: self.scene.setup_parametric_interpolation()
        elif symbol == key.W: self.scene.where()
        elif symbol == key.U: self.usage()
        elif symbol == key.L: self.scene.toggle("line")
        elif symbol == key.I: self.scene.toggle("fill")
        elif symbol == key.T: self.scene.toggle("transparent")
        elif symbol == key.P: self.scene.camera.toggle("parallel")
        elif symbol == key.G: self.scene.toggle("light")
        elif symbol == key.O: self.scene.toggle("drawsolid")
        elif symbol == key.K: self.scene.toggle("markers")
        elif symbol == key.EXCLAMATION: self.scene.toggle("photonmagic")
        elif symbol == key.M: self.toggle_animate()
        elif symbol == key.C: self.scene.toggle_cuda()
        elif symbol == key.R: self.scene.toggle_raycast()
        elif symbol == key.J: self.scene.toggle_showmetric()
        elif symbol == key.E: self.save_to_file()
        elif symbol == key.D: self.scene.step()
        elif symbol == key.AT: self.scene.clipper.reset()
        elif symbol == key.POUND: self.scene.reload_()
        elif symbol == key.BRACKETRIGHT: self.scene.loadnext()
        elif symbol == key.BRACKETLEFT: self.scene.loadprev()
        elif symbol == key.TAB: 
            self.tab_mode = True
            self.scene.visit_bookmark(None)
        elif symbol in number_keys:
            self.bookmark_mode = True
            self.bookmark_key = symbol - key._0
        elif symbol is None:
            log.warn("on_key_press getting None symbol modifiers %s " % modifiers )
        else:
            pass
            print "no action for on_key_press with symbol 0x%x %s modifiers %s " % ( symbol, key.symbol_string(symbol), modifiers )
        pass
        if self.dragmode:

            modfactor = 1
            if modifiers & key.MOD_SHIFT:
               modfactor *= 2
            if modifiers & key.MOD_ALT:
               modfactor *= 2

            self.modfactor = modfactor
            #print "dragmode modifiers %s modfactor %s " % (key.modifiers_string(modifiers), self.modfactor)
 
        self.redraw()

    def on_key_release(self,symbol, modifiers):
        if   symbol == key.Z: self.zoom_mode = False
        elif symbol == key.QUOTELEFT: self.time_mode = False
        elif symbol == key.A: self.scan_mode = False
        elif symbol == key.X: self.pan_mode = False
        elif symbol == key.N: self.near_mode = False
        elif symbol == key.F: self.far_mode = False
        elif symbol == key.Y: self.yfov_mode = False
        elif symbol == key.Q: self.target_mode = False
        elif symbol == key.TAB: 
            self.tab_mode = False
        elif symbol in number_keys:
            self.bookmark_mode = False
            if self.bookmark_key is None: 
                pass  # means just created a bookmark by virtue of mouse press, not jumping to it 
            else:
                self.scene.visit_bookmark(self.bookmark_key)
            pass
        else:
            pass
            #print "no action for on_key_release with symbol 0x%x " % symbol
        pass
        self.modfactor = 1
        self.redraw()


    def toggle_animate(self):
        """
        Invoked by pressing M, the effect depends on whether 
        in time_mode (holding QUOTELEFT).

        #. not time_mode: viewpoint animation
        #. time_mode: event propagation animation

        TODO:

        #. arrange event propagation animation to pick up from the current interactively 
           set global time rather than starting over from zero 

        """
        if not self.time_mode:
            self.scene.toggle_animate()
        else:
            self.scene.event.toggle_animate()

    def animation_period(self, factor):
        if not self.time_mode:
            self.scene.animation_period(factor)
        else:
            self.scene.event.animation_period(factor)

    def on_mouse_drag(self,_x,_y,_dx,_dy,button):

        width = float(self.viewport.width)
        height = float(self.viewport.height)
        dragfactor = self.modfactor * self.dragfactor 

        x  = dragfactor*(_x*2.0 - width)/width
        dx = dragfactor*(2.*_dx)/width

        y  = dragfactor*(_y*2.0 - height)/height
        dy = dragfactor*(2.*_dy)/height

        #log.info("on_mouse_drag x %s y %s dx %s dy %s dragfactor %s " % (x,y,dx,dy, dragfactor ))

        two_finger_zoom = button == 8    # NB zoom is a misnomer, this is translating eye coordinate z
        if   self.zoom_mode or two_finger_zoom: self.scene.trackball.zoom_to(x,y,dx,dy)
        elif self.scan_mode: self.scene.event.scan_to(x,y,dx,dy)  # used for qcut for interactive photon selection, primitive time cut
        elif self.time_mode: self.scene.event.time_to(x,y,dx,dy)
        elif self.pan_mode: self.scene.trackball.pan_to(x,y,dx,dy)
        elif self.near_mode: self.scene.camera.near_to(x,y,dx,dy)
        elif self.far_mode: self.scene.camera.far_to(x,y,dx,dy)
        elif self.yfov_mode: self.scene.camera.yfov_to(x,y,dx,dy)
        else: 
            self.scene.trackball.drag_to(x,y,dx,dy)  # default is to rotate
        pass
        self.redraw()

    def on_mouse_press(self, x, y, button):
        #log.info("on_mouse_press %d " % button)
        if button != 2:print 'Mouse button pressed (x=%.1f, y=%.1f, button=%d)' % (x,y,button)
        xyz = self.frame_handler.unproject(x,y)
        if xyz is None:
            log.warn("on_mouse_press unprojection failed ") 
            return

        if self.bookmark_mode:
            self.scene.create_bookmark( xyz, self.bookmark_key )
            self.bookmark_key = None # signal that bookmark was created, thanks to the mouse press 
        else:
            self.scene.clicked_point( xyz, self.target_mode )
        pass
        self.redraw()

    def on_mouse_release(self, x, y, button):
        pass
        if button != 2:print 'Mouse button released (x=%.1f, y=%.1f, button=%d)' % (x,y,button)

    def on_mouse_motion(self, x, y, dx, dy):
        pass

    def on_mouse_scroll(self, x, y, dx, dy):
        print 'Mouse scroll (x=%.1f, y=%.1f, dx=%.1f, dy=%.1f)' % (x,y,dx,dy)   # none of these

    def on_idle(self, dt):
        """
        Hmm giving crazy high fps, because normally no updates/redrawing happen
        """
        global t, t0, frames
        t += dt
        frames = frames + 1 
        if t-t0 > 5.0:
            fps = float(frames)/(t-t0)
            self.fps = fps
            #print 'FPS: %.2f (%d frames in %.2f seconds)' % (fps, frames, t-t0)
            frames,t0 = 0, t
        pass
        self.frame_handler.tick(dt)


if __name__ == '__main__':
    pass


