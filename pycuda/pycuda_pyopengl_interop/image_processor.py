#!/usr/bin/env python
import ctypes
import numpy as np
import OpenGL.GL as gl
import OpenGL.raw.GL.VERSION.GL_1_1 as rawgl   #initially OpenGL.raw.GL as rawgl but only GL_1_1 has the glReadPixels symbol
import pycuda.gl as cuda_gl
import pycuda.driver as cuda_driver

from pixel_buffer import PixelBuffer
from gpu.tools import get_cu_module, cuda_options, GPUFuncs, get_cu_source


class ImageProcessor(object):
    def __init__(self, w, h ):
        self.resize(w, h)
        self.dest   = PixelBuffer(w, h, texture=True) 
        self.cuda_init()

    def resize(self, w,h ):
        self.image_width, self.image_height = w,h

    def cuda_init(self):
        raise Exception("sub classes expected to implement this")

    def cuda_process(self):
        raise Exception("sub classes expected to implement this")

    def display(self,*args,**kwa):
        self.dest.draw(*args,**kwa)

    def cleanup(self):
        self.dest.cleanup()




class ImageFilter(ImageProcessor):
    def __init__(self, w,h  ):
        ImageProcessor.__init__(self, w, h)
        self.source = PixelBuffer(w, h) 

    def process(self,*args,**kwa):
        self.source.load_from_framebuffer(*args,**kwa)
        self.cuda_process()

    def cleanup(self):
        self.source.cleanup()
        self.dest.cleanup()




class ImageGenerator(ImageProcessor):
    def __init__(self, w,h ):
        ImageProcessor.__init__(self, w, h)
 
    def process(self,*args,**kwa):
        self.cuda_process()



class Invert(ImageFilter):

    def cuda_init(self):
        invert = GPUFuncs(get_cu_module("invert.cu", options=cuda_options)).invert
        invert.prepare("PP")  # kernel takes two PBOs as arguments
        self.invert = invert

    def cuda_process(self):

        N = 16                # 32 also works at **same fps**, 32*32=1024 is the max threads per block on NVIDIA 750M
        grid = (self.image_width//N,self.image_height//N)
        block = (N, N, 1)        # threads per block 

        source_mapping = self.source.cuda_pbo.map()
        dest_mapping   = self.dest.cuda_pbo.map()

        self.invert.prepared_call(grid, block, source_mapping.device_ptr(), dest_mapping.device_ptr())
        cuda_driver.Context.synchronize()

        source_mapping.unmap()
        dest_mapping.unmap()



class Generate(ImageGenerator):

    def cuda_init(self):
        generate = GPUFuncs(get_cu_module("invert.cu", options=cuda_options)).generate
        generate.prepare("P")  
        self.generate = generate

    def cuda_process(self):

        N = 16                # 32 also works at **same fps**, 32*32=1024 is the max threads per block on NVIDIA 750M
        grid = (self.image_width//N,self.image_height//N)
        block = (N, N, 1)        # threads per block 

        dest_mapping   = self.dest.cuda_pbo.map()
        self.generate.prepared_call(grid, block, dest_mapping.device_ptr())
        cuda_driver.Context.synchronize()
        dest_mapping.unmap()





if __name__ == '__main__':
    pass

