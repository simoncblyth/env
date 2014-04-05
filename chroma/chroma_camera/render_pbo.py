#!/usr/bin/env python
"""
::

    python render_pbo.py -s 1024,768 -k render_pbo_debug


"""
import os, argparse, logging
log = logging.getLogger(__name__)
from collections import OrderedDict

import numpy as np
import glumpy as gp

import pycuda.gl as cuda_gl
import pycuda.driver as cuda
import pycuda.gpuarray as ga
import pycuda.gl.autoinit

from chroma import gpu
from chroma.gpu.tools import get_cu_module, cuda_options, GPUFuncs, chunk_iterator
from chroma.loader import load_geometry_from_string

from env.pycuda.pycuda_pyopengl_interop.pixel_buffer import PixelBuffer


class Camera(object):
    def __init__(self, geometry, pixels, config):
        self.renderer = GPURenderer( pixels, geometry, config )

    def init(self):
        pass

    def render_pixels(self):
        self.renderer.render()


class LaunchSequence(object):
    def __init__(self, worksize, max_blocks=1024, threads_per_block=64 ):
        self.worksize = worksize
        self.max_blocks = max_blocks
        self.threads_per_block = threads_per_block
        pass

    chunker = property(lambda self:chunk_iterator(self.worksize, self.threads_per_block, self.max_blocks))
    counts = property(lambda self:[_[1] for _ in self.chunker])
    block = property(lambda self:(self.threads_per_block,1,1))

    def __str__(self):
        def present_launch((offset, count, blocks_per_grid)):
            grid=(blocks_per_grid, 1)
            block=(self.threads_per_block,1,1)
            return "offset %10s count %s grid %s block %s " % ( offset, count, repr(grid), repr(block) )
        pass
        return "\n".join([self.smry]+map(present_launch, self.chunker))

    def _get_smry(self):
        counts = self.counts
        assert sum(counts) == self.worksize
        return "%s worksize %s max_blocks %s threads_per_block %s launches %s " % (self.__class__.__name__, self.worksize, self.max_blocks, self.threads_per_block, len(counts))
    smry = property(_get_smry)

    def __repr__(self):
        return self.smry



class GPURenderer(object):
    def __init__(self, pixels, geometry, config, origin=None , pixel2world=None ):
        pass
        npixels = pixels.npixels
        size = pixels.size

        self.config = config
        self.pixels = pixels
        self.gpu_geometry = gpu.GPUGeometry( geometry )
        self.npixels = npixels
        launch = LaunchSequence(npixels, threads_per_block=config.args.threads_per_block, max_blocks=config.args.max_blocks )
        self.launch = launch
        #print launch 

        if origin is None:
            origin = np.array([200,150,150,1])
        if pixel2world is None:
            pixel2world = np.identity(4)

        dx_size = config.args.max_alpha_depth*npixels
        self.dx    = ga.empty(dx_size, dtype=np.float32)
        self.dxlen = ga.zeros(npixels, dtype=np.uint32)
        self.color = ga.empty(npixels, dtype=ga.vec.float4)

        self.compile_()
        self.set_constants(size, origin, pixel2world)

    def compile_(self):
        """
        #. compile kernel and extract __constant__ symbol addresses
        """
        module = get_cu_module('render_pbo.cu', options=cuda_options)
        self.arg_format = "iiPPPPP"

        self.g_size   = module.get_global("g_size")[0]  
        self.g_origin = module.get_global("g_origin")[0]
        self.g_pixel2world = module.get_global("g_pixel2world")[0]  

        kernel = module.get_function(self.config.args.kernel)
        kernel.prepare(self.arg_format)
        self.kernel = kernel

    def set_constants(self, size, origin, pixel2world ):
        """ copy constant values to GPU """
        cuda.memcpy_htod(self.g_size,         ga.vec.make_int2(*size))
        cuda.memcpy_htod(self.g_origin,       ga.vec.make_float4(*origin))
        cuda.memcpy_htod(self.g_pixel2world,  np.float32(pixel2world))

    def check_args(self, arg_format, *args ):
        from pycuda._pvt_struct import pack
        i = 0
        for fmt,arg in zip(arg_format,args):
            print "checking arg %s %s %s " % (i,fmt, arg) 
            arg_buf = pack(fmt,arg)
            i += 1

    def render(self, alpha_depth=3, keep_last_render=False, check_args=False):
        """
        :param alpha_depth:
        :param keep_last_render:

        * http://stackoverflow.com/questions/6954487/how-to-use-the-prepare-function-from-pycuda

        """
        assert alpha_depth <= self.config.args.max_alpha_depth
        if not keep_last_render:
            self.dxlen.fill(0)   # this is calling a gpuarray.fill kernel 

        pbo_mapping = self.pixels.cuda_pbo.map()

        args = [ np.uint32(alpha_depth), 
                 pbo_mapping.device_ptr(),
                 self.gpu_geometry.gpudata, 
                 self.dx.gpudata, 
                 self.dxlen.gpudata, 
                 self.color.gpudata ]

        if check_args:
            self.check_args( self.arg_format[1:], *args)  # skip offset arg

        block = self.launch.block
        calls = 0 
        print "pixels %s launch %s " % (repr(self.pixels.size), repr(self.launch))
        for offset, count, blocks_per_grid in self.launch.chunker:
            grid=(blocks_per_grid, 1)
            print "[%s] offset %s grid %s block %s " % (calls, offset, repr(grid),repr(block))
            self.kernel.prepared_call( grid, block, np.uint32(offset), *args )
            calls += 1
        pass
        cuda.Context.synchronize()  # OMITTING THIS SYNC CAUSES AN UNRECOVERABLE GUI FREEZE
        pbo_mapping.unmap()


class FigHandler(object):
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
    def __init__(self, frame, scene ):
        frame.push(self)
        self.frame = frame
        self.scene = scene

    def on_draw(self):
        self.frame.lock()
        self.frame.draw()
        self.scene.draw()
        self.frame.unlock()


class Scene(object):
    def __init__(self, camera, trackball, pixels):
        self.camera = camera
        self.trackball = trackball
        self.pixels = pixels

    def init(self):
        self.camera.init()

    def draw(self):
        self.trackball.push()
        self.camera.render_pixels()
        self.pixels.draw()
        self.trackball.pop()



class Config(object):
    def __init__(self, doc):
        parser, defaults = self._make_parser(doc)
        self.parser = parser
        self.defaults = defaults
        self.parse()

    def parse(self):
        args = self.parser.parse_args()
        logging.basicConfig(level=getattr(logging, args.loglevel), format="%(asctime)-15s %(message)s")
        np.set_printoptions(precision=4, suppress=True)
        self.args = args
 
    def _make_parser(self, doc):
        parser = argparse.ArgumentParser(doc)

        defaults = OrderedDict()
        defaults['loglevel'] = "INFO"
        defaults['threads_per_block'] = 64
        defaults['max_blocks'] = 128
        defaults['max_alpha_depth'] = 3
        defaults['size'] = "200,200"
        defaults['kernel'] = "render_pbo"

        parser.add_argument("-l","--loglevel",help="INFO/DEBUG/WARN/..   %(default)s")  
        parser.add_argument( "-t","--threads-per-block", help="", type=int  )
        parser.add_argument( "-a","--max-alpha-depth", help="", type=int  )
        parser.add_argument( "-b","--max-blocks", help="", type=int  )
        parser.add_argument( "-s","--size", help="", type=str  )
        parser.add_argument( "-k","--kernel", help="", type=str  )

        parser.set_defaults(**defaults)
        return parser, defaults

    size=property(lambda self:map(int,self.args.size.split(",")))

        
    def _settings(self, args, defaults, all=False):
        if args is None:return "PARSE ERROR"
        if all:
            filter_ = lambda kv:True
        else:
            filter_ = lambda kv:kv[1] != getattr(args,kv[0]) 
        pass
        wid = 20
        fmt = " %-15s : %20s : %20s "
        return "\n".join([ fmt % (k,str(v)[:wid],str(getattr(args,k))[:wid]) for k,v in filter(filter_,defaults.items()) ])

    def settings(self, all_=False):
        return self._settings( self.args, self.defaults, all_ )

    def all_settings(self):
        return "\n".join(filter(None,[
                      self.settings(True) ,
                         ]))
    def changed_settings(self):
        return "\n".join(filter(None,[
                      self.settings(False) ,
                         ]))
    def __repr__(self):
        return self.all_settings() 
 


def main():
    """
    Larger sizes lead to failures after the last kernel launch in the sequence
    """

    config = Config(__doc__)
    print config

    fig = gp.figure(config.size)
    frame = fig.add_frame()

    geometry = load_geometry_from_string(os.environ['DAE_NAME'])
    pixels = PixelBuffer(config.size, texture=True)

    camera = Camera(geometry, pixels, config)
    trackball = gp.Trackball( 65, 135, 1.0, 2.5 )

    scene = Scene(camera, trackball, pixels)
    fighandler = FigHandler(fig, scene)
    framehandler = FrameHandler(frame, scene)

    gp.show()





if __name__ == '__main__':
    main()

