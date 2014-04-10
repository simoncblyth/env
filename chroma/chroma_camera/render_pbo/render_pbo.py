#!/usr/bin/env python
"""
::

    ./render_pbo.py -s 1024,768 -k render_pbo_debug
    ./render_pbo.py --cuda-profile --alpha-depth 10
    ./render_pbo.py --cuda-profile --alpha-depth 10 --kernel render_pbo --size 1024,768 --view B --kernel-flags 18,0

    ./render_pbo.py -s 1024,768 -k render_pbo --max-time 1


NB to avoid seeing just noise when looking at metrics, first bit shift to black then 
scan downwards to look for structure/hotspots:: 

    ./render_pbo.py -s 1024,768 -k render_pbo --max-time 1 --metric time --kernel-flags 24,0    # this leaves a black screen
    ./render_pbo.py -s 1024,768 -k render_pbo --max-time 1 --metric time --kernel-flags 19,0    # dial down see the hotspots

    ./render_pbo.py -s 1024,768 -k render_pbo --max-time 1 --metric node --kernel-flags 2,0


"""
import logging
log = logging.getLogger(__name__)

import glumpy as gp

from env.cuda.cuda_launch import CUDACheck
from render_pbo_config import Config

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
    config.cudacheck = CUDACheck(config)  # MUST be done before CUDA init, for setting of CUDA_PROFILE envvar 

    from render_pbo_scene import Scene

    fig = gp.figure(config.size)  # glumpy glut setup
    frame = fig.add_frame((1,1))
    trackball = gp.Trackball( 65, 135, 1.0, 2.5 )

    scene = Scene(config, trackball)       

    fighandler = FigHandler(fig, scene)    # glumpy event handlers
    framehandler = FrameHandler(frame, scene)

    gp.show()



if __name__ == '__main__':
    main()



