#!/usr/bin/env python
"""

Controls
---------

* SPACE and mouse/trackpad up/down to translate z (in-out of direction of view)
* click+two-finger draw, a non-modal alternative for z translation
* TAB and mouse/trackpad to translate x,y (left-right,up-down)
* **S** toggle fullscreen 
* **F** toggle fill
* **L** toggle line
* **T** toggle transparent
* **P** toggle parallel projection (aka orthographic)
* **N** toggle near mode, where mouse/trackpad changes near clipping plane 
* **A** toggle far mode, where mouse/trackpad changes far clipping plane 
* **Y** toggle yfov mode, where mouse/trackpad changes field of view 
* **O** toggle clicked solid marker
* **UP** and **DOWN** arrow changes trackball translation scale factor
* **H** trackball home, reset trackball translation and rotation offsets to zero
* **M** start interpolation


"""
import sys, logging
log = logging.getLogger(__name__)
from glumpy.window import key

from daedispatcher import DAEDispatcher
from daeviewport import DAEViewport


class DAEInteractivityHandler(object):
    """
    Keep this for handling interactivity, **NOT** graphics, **NOT state**     
    """
    def __init__(self, fig, frame_handler, scene, config ):
        #
        self.fig = fig
        self.frame_handler = frame_handler
        self.dragfactor = config.args.dragfactor
        self.scene = scene
        self.viewport = DAEViewport(map(int,config.args.size.split(",")))
        #
        self.zoom_mode = False
        self.pan_mode = False
        self.near_mode = False
        self.far_mode = False
        self.yfov_mode = False
        #
        if config.args.fullscreen:
            self.toggle_fullscreen()
        pass

        fig.push(self)                     # event notification from fig 
        self.hookup_dispatcher(config)     # event notification from dispatcher


    def __repr__(self):
        return "H %5.2f " % self.dragfactor

    def hookup_dispatcher(self, config):
        """
        dispatcher collect messages over UDP, allowing remote control
        """
        dispatcher = DAEDispatcher(port=config.args.port, host=config.args.host)
        log.info(dispatcher)        

        def _check_dispatcher(dt):
            dispatcher.update()
        timer = self.fig.timer(30.)  # fps
        timer(_check_dispatcher) 
        dispatcher.push_handlers(self)   # get event notification from dispatcher

    def _get_title(self):
        return " ".join(map(repr,[
                     self.scene,
                     self.scene.view,
                     self.scene.camera,
                     self.frame_handler,
                     self.scene.trackball,
                     self,
                     ]))
    title = property(_get_title)     

    def redraw(self):
        self.fig.window.set_title(self.title)
        self.fig.redraw()

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

    def on_resize(self, width, height, x=0, y=0):
        self.viewport.resize((width, height))
        self.scene.resize((width, height))

    def on_draw(self):
        self.fig.clear(0.85,0.85,0.85,1)  # seems to have no effect even when lighting disabled

    def on_key_press(self, symbol, modifiers):
        if   symbol == key.ESCAPE: sys.exit();
        elif symbol == key.SPACE: self.zoom_mode = True
        elif symbol == key.TAB: self.pan_mode = True
        elif symbol == key.N: self.near_mode = True
        elif symbol == key.A: self.far_mode = True
        elif symbol == key.Y: self.yfov_mode = True
        elif symbol == key.UP: self.dragfactor *= 2.
        elif symbol == key.DOWN: self.dragfactor *= 0.5
        elif symbol == key.S: self.toggle_fullscreen()
        elif symbol == key.LEFT: self.scene.animation_speed(0.5)
        elif symbol == key.RIGHT: self.scene.animation_speed(2.0)
        elif symbol == key.H: self.scene.trackball.home()
        elif symbol == key.W: self.scene.where()
        elif symbol == key.B: self.scene.bookmark()
        elif symbol == key.L: self.scene.toggle_attr("line")
        elif symbol == key.F: self.scene.toggle_attr("fill")
        elif symbol == key.T: self.scene.toggle_attr("transparent")
        elif symbol == key.P: self.scene.toggle_attr("parallel")
        elif symbol == key.M: self.scene.toggle_attr("animate")
        elif symbol == key.G: self.scene.toggle_attr("light")
        elif symbol == key.O: self.scene.toggle_attr("drawsolid")
        elif symbol == key.C: self.scene.toggle_attr("cuda")
        elif symbol == key.K: self.scene.toggle_attr("markers")
        elif symbol == key.R: self.scene.toggle_attr("raycast")
        else:
            pass
            #print "no action for on_key_press with symbol 0x%x " % symbol
        pass 
        self.redraw()

    def on_key_release(self,symbol, modifiers):
        if   symbol == key.SPACE: self.zoom_mode = False
        elif symbol == key.TAB: self.pan_mode = False
        elif symbol == key.N: self.near_mode = False
        elif symbol == key.A: self.far_mode = False
        elif symbol == key.Y: self.yfov_mode = False
        else:
            pass
            #print "no action for on_key_release with symbol 0x%x " % symbol
        pass
        self.redraw()

    def on_mouse_drag(self,_x,_y,_dx,_dy,button):

        width = float(self.viewport.width)
        height = float(self.viewport.height)
        dragfactor = self.dragfactor 

        x  = dragfactor*(_x*2.0 - width)/width
        dx = dragfactor*(2.*_dx)/width

        y  = dragfactor*(_y*2.0 - height)/height
        dy = dragfactor*(2.*_dy)/height

        #log.info("on_mouse_drag x %s y %s dx %s dy %s dragfactor %s " % (x,y,dx,dy, dragfactor ))

        two_finger_zoom = button == 8    # NB zoom is a misnomer, this is translating eye coordinate z
        if   self.zoom_mode or two_finger_zoom: self.scene.trackball.zoom_to(x,y,dx,dy)
        elif self.pan_mode: self.scene.trackball.pan_to(x,y,dx,dy)
        elif self.near_mode: self.scene.camera.near_to(x,y,dx,dy)
        elif self.far_mode: self.scene.camera.far_to(x,y,dx,dy)
        elif self.yfov_mode: self.scene.camera.yfov_to(x,y,dx,dy)
        else: 
            self.scene.trackball.drag_to(x,y,dx,dy)  # default is to rotate
        pass
        self.redraw()

    def on_mouse_press(self, x, y, button):
        if button != 2:print 'Mouse button pressed (x=%.1f, y=%.1f, button=%d)' % (x,y,button)
        xyz = self.frame_handler.unproject(x,y)
        self.scene.clicked_point( xyz )
        self.redraw()

    def on_mouse_release(self, x, y, button):
        pass
        if button != 2:print 'Mouse button released (x=%.1f, y=%.1f, button=%d)' % (x,y,button)

    def on_mouse_motion(self, x, y, dx, dy):
        pass

    def on_mouse_scroll(self, x, y, dx, dy):
        print 'Mouse scroll (x=%.1f, y=%.1f, dx=%.1f, dy=%.1f)' % (x,y,dx,dy)   # none of these

    def on_idle(self, dt):
        self.frame_handler.tick(dt)



if __name__ == '__main__':
    pass


