#!/usr/bin/env python
"""

Controls
---------

* SPACE and mouse/trackpad up/down to translate z (in-out of direction of view), 
  also a non-modal alternative is click+two-finger drag 
* TAB and mouse/trackpad to translate x,y (left-right,up-down)
* **S** toggle fullscreen 
* **F** toggle fill
* **L** toggle line
* **T** toggle transparent
* **P** toggle parallel projection (aka orthographic)
* **N** toggle near mode, where mouse/trackpad changes near clipping plane 
* **A** toggle far mode, where mouse/trackpad changes far clipping plane 
* **Y** toggle yfov mode, where mouse/trackpad changes field of view 
  also **UP** and **DOWN** arrow yets changes yfov in 5 degrees increments within
  range of 5 to 175 degrees. Extreme wideangle is useful when using parallel projection. 


"""
import sys
from glumpy.window import key


class DAEInteractivityHandler(object):
    """
    Keep this for handling interactivity, **NOT** graphics    
    """
    def __init__(self, fig, frame_handler, scene, config ):
        self.fig = fig
        self.frame_handler = frame_handler
        self.scene = scene
        self.trackball = scene.trackball
        self.view = scene.view
        # 
        self.zoom_mode = False
        self.pan_mode = False
        self.near_mode = False
        self.far_mode = False
        self.yfov_mode = False

        if config.args.fullscreen:
            self.toggle_fullscreen()
        pass
        fig.push(self)   # event notification

    def _get_title(self):
        return "%s %s %s" % (repr(self.frame_handler),repr(self.view),repr(self.trackball))
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

    def on_draw(self):
        self.fig.clear(0.85,0.85,0.85,1)  # seems to have no effect even when lighting disabled

    def on_key_press(self, symbol, modifiers):
        if   symbol == key.ESCAPE: sys.exit();
        elif symbol == key.SPACE: self.zoom_mode = True
        elif symbol == key.TAB: self.pan_mode = True
        elif symbol == key.N: self.near_mode = True
        elif symbol == key.A: self.far_mode = True
        elif symbol == key.Y: self.yfov_mode = True
        elif symbol == key.UP: self.frame_handler.tweak_scale(2.0)
        elif symbol == key.DOWN: self.frame_handler.tweak_scale(0.5)
        elif symbol == key.S: self.toggle_fullscreen()
        elif symbol == key.L: self.frame_handler.toggle_line()
        elif symbol == key.F: self.frame_handler.toggle_fill()
        elif symbol == key.T: self.frame_handler.toggle_transparent()
        elif symbol == key.P: self.frame_handler.toggle_parallel()
        elif symbol == key.M: self.frame_handler.toggle_animate()
        else:
            print "no action for on_key_press with symbol 0x%x " % symbol
        pass 
        self.redraw()

    def on_key_release(self,symbol, modifiers):
        if   symbol == key.SPACE: self.zoom_mode = False
        elif symbol == key.TAB: self.pan_mode = False
        elif symbol == key.N: self.near_mode = False
        elif symbol == key.A: self.far_mode = False
        elif symbol == key.Y: self.yfov_mode = False
        else:
            print "no action for on_key_release with symbol 0x%x " % symbol
        pass
        self.redraw()

    def on_mouse_drag(self,x,y,dx,dy,button):
        two_finger_zoom = button == 8  # NB zoom is a misnomer, this is translating eye coordinate z
        if   self.zoom_mode or two_finger_zoom: self.trackball.zoom_to(x,y,dx,dy)
        elif self.pan_mode: self.trackball.pan_to(x,y,dx,dy)
        elif self.near_mode: self.trackball.near_to(x,y,dx,dy)
        elif self.far_mode: self.trackball.far_to(x,y,dx,dy)
        elif self.yfov_mode: self.trackball.yfov_to(x,y,dx,dy)
        else: 
            self.trackball.drag_to(x,y,dx,dy)
        pass
        self.redraw()

    def on_mouse_press(self, x, y, button):
        print 'Mouse button pressed (x=%.1f, y=%.1f, button=%d)' % (x,y,button)

    def on_mouse_release(self, x, y, button):
        print 'Mouse button released (x=%.1f, y=%.1f, button=%d)' % (x,y,button)

    def on_mouse_motion(self, x, y, dx, dy):
        pass
        #print 'Mouse motion (x=%.1f, y=%.1f, dx=%.1f, dy=%.1f)' % (x,y,dx,dy)  # get lots of these

    def on_mouse_scroll(self, x, y, dx, dy):
        print 'Mouse scroll (x=%.1f, y=%.1f, dx=%.1f, dy=%.1f)' % (x,y,dx,dy)   # none of these

    def on_idle(self, dt):
        self.frame_handler.tick(dt)



if __name__ == '__main__':
    pass


