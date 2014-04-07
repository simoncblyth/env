#!/usr/bin/env python
"""
::

    ./render_pbo.py -s 1024,768 -k render_pbo_debug
    ./render_pbo.py --cuda-profile --alpha-depth 10


"""
import logging
log = logging.getLogger(__name__)

import glumpy as gp

from render_pbo_config import Config
from env.cuda.cuda_launch import CUDACheck

class FigHandler(object):
    """glumpy event handler"""
    def __init__(self, fig, scene):
        fig.push(self)
        self.fig = fig 
        self.scene = scene
    def on_init(self):
        self.scene.init()
    def on_mouse_drag(self, x,y,dx,dy,button):
        self.scene.trackball.drag_to(x,y,dx,dy)
        self.fig.redraw()
    def on_draw(self):
        self.fig.clear(0.85,0.85,0.85,1)

class FrameHandler(object):
    """glumpy event handler"""
    def __init__(self, frame, scene ):
        frame.push(self)
        self.frame = frame
        self.scene = scene
    def on_draw(self):
        self.frame.lock()
        self.frame.draw()
        self.scene.draw()
        self.frame.unlock()


def main():
    config = Config(__doc__)
    print config
    cudacheck = CUDACheck(config)

    log.info("render_pbo main")
    from render_pbo_scene import Scene


    fig = gp.figure(config.size)  # glumpy glut setup
    frame = fig.add_frame()
    trackball = gp.Trackball( 65, 135, 1.0, 2.5 )

    scene = Scene(config, trackball)       

    fighandler = FigHandler(fig, scene)    # glumpy event handlers
    framehandler = FrameHandler(frame, scene)

    gp.show()

    cudacheck.tail()


if __name__ == '__main__':
    main()



