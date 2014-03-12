#!/usr/bin/env python
"""
::

   ./simplecamera.py -s1 --eye=-.5,-.5,0 -d3    

   ./simplecamera.py -s3155 -d3 

        # http://belle7.nuu.edu.tw/dae/tree/3155___1.html
        # complicated full AD geometry 
        # sometimes causes kernel GPU panic

    ./simplecamera.py -s3155 -d3 -f10 --eye=0.01,0.01,0.01 --lookat=0,0,-1   # down
    ./simplecamera.py -s3155 -d3 -f10 --eye=0.01,0.01,0.01 --lookat=0,0,1    # up
    ./simplecamera.py -s3155 -d3 -f10 --eye=0.01,0.01,0.01 --lookat=0,1,0   # side

        # near center of an AD looking up/down/side no PMTs visible

    ./simplecamera.py -s3155 -d10 -f10 --eye=0.01,0.01,0.01 --lookat=0,1,0

       # still no visible PMT even with alpha depth at max 

       * http://belle7.nuu.edu.tw/dae/tree/3155___1.html?cam=0.01,0.01,0.01&look=0,1,0

         orientation mis-match ?

   ./simplecamera.py -s3199 -d3 -f20 --eye=0,-1,0    # from outside radial shield
   ./simplecamera.py -s3199 -d3 -f20 --eye=0,1,0     # from inside
        # ray traces targetting a single PMT insitu

   ./simplecamera.py -s3199 -d3 -f10 --eye=0,1,0 --lookat=10,0,10  
        # from inside, look around, whacky angle many PMTs sticking thru shield


Options.

`-s1`
     solid to present by index
`-d3` 
     depth 3 is one more alpha layer than default 
`-f20`
     focal length in mm, corresponding to 35mm film


Aligning with daeserver, so can line things up with 
fast and reliable webgl daeserver before try to chroma ray trace
        

"""
import numpy as np
import os

import pycuda.driver as cuda
from pycuda import gpuarray as ga

from chroma.log import logger, logging
logger.setLevel(logging.INFO)

from chroma.transform import rotate
from chroma.tools import from_film
from chroma import gpu
from chroma.loader import load_geometry_from_string


import pygame
from pygame.locals import *

from myarcball import MyArcball

def fromstring( xyz, n=3 ):
    a = np.fromstring(xyz, sep=",")
    if len(a) == n:
        return a
    elif len(a) > n:
        logger.info("unpacked too many %s " % a )
        assert 0
    elif len(a) == 1 and n == 3:
        return np.array([a[0],a[0],a[0]])
    elif len(a) == 1 and n == 2:
        return np.array([a[0],a[0]])
    else:
        assert 0      


class SimpleCamera(object):
    "The camera class is used to render a Geometry object."
    def __init__(self, geometry, size="800,600", device_id=None, solid=None, eye="0,1,0", lookat="0,0,0", up="-1,0,0", alpha_max=3, focal_length=55, radius=0.5, headless=False, savepath=None ):
        super(SimpleCamera, self).__init__()
        logger.info("Camera.__init__")
        self.geometry = geometry
        self.device_id = device_id
        self.solid_index = solid

        self.eye = fromstring(eye,n=3)
        self.lookat = fromstring(lookat,n=3)
        self.up = fromstring(up,n=3)

        self.size = map(int, size.split(","))
        self.width, self.height = self.size

        self.step = 1000
        self.max_alpha_depth = alpha_max
        self.alpha_depth = self.max_alpha_depth
        self.film_width = 35.0 # mm
        self.focal_length = focal_length # mm
        self.radius = radius
        self.headless = headless
        self.savepath = savepath

        if headless:
            os.environ['SDL_VIDEODRIVER'] = 'dummy'


    def parameter_summary(self):
         # hmm, duplication between here and argparser 
         items = [
             ["-s/--solid", self.solid_index],
             ["-e/--eye",  self.eye],
             ["-a/--lookat", self.lookat],
             ["-u/--up", self.up],
             ["-r/--size", self.size],
             ["-d/--alpha-max", self.max_alpha_depth],
             ["-f/--focal-length", self.focal_length],
             ["-r/--radius", self.radius],
             ["-G/--headless", self.headless],
             ["-o/--savepath", self.savepath],
                 ]
         return "\n".join(["%-20s : %s " % (k,v) for k,v in items]) 

    def init_camera(self):
        if self.solid_index is None:
            bounds = self.geometry.mesh.get_bounds()
            logger.info("using global bounds %s " % (str(bounds)))
        else:
            solid_index = int(self.solid_index)
            bounds = self.geometry.solids[solid_index].mesh.get_bounds()
            logger.info("using bounds %s for solid_index %s " % (str(bounds), solid_index)) 

        lower_bound, upper_bound = bounds
        center = np.mean( bounds, axis=0 )
        extent = np.linalg.norm(upper_bound-lower_bound)

        punit = extent  # unit of the input parameters to convert to world coordinates
        lookat = center + self.lookat*punit 
        eye  = center + self.eye*punit 

        forward = lookat - eye 
        distance = np.linalg.norm(forward)
        forward /= distance 

        world_up = self.up
        left   = np.cross( world_up, forward ) 
        cam_up = np.cross( forward, left )
       
        print "cam_up (axis2)   %s " % cam_up
        print "left   (axis1)   %s " % left 
        print "forward          %s " % forward 

        # world dimensions, coordinates
        self.diagonal = extent
        self.scale = extent
        self.point = eye  

        # axis1 along height of film, axis2 along width of film (or vv)
        self.axis2 = cam_up
        self.axis1 = left
        self.ball = MyArcball.make( self.size, self.axis1, self.axis2, constrain=True, radius=self.radius )



    def init_gpu(self):
        self.context = gpu.create_cuda_context(self.device_id)
        self.gpu_geometry = gpu.GPUGeometry(self.geometry)
        self.gpu_funcs = gpu.GPUFuncs(gpu.get_cu_module('mesh.h'))
        self.npixels = self.width*self.height
        self.clock = pygame.time.Clock()

        pos, dir = from_film(self.point, axis1=self.axis1, axis2=self.axis2,
                             size=self.size, width=self.film_width, focal_length=self.focal_length)

        self.rays = gpu.GPURays(pos, dir, max_alpha_depth=self.max_alpha_depth)

        self.pixels_gpu = ga.empty(self.npixels, dtype=np.uint32)


    def rotate(self, phi, n):
        self.rays.rotate(phi, n)

        self.point = rotate(self.point, phi, n)
        self.axis1 = rotate(self.axis1, phi, n)
        self.axis2 = rotate(self.axis2, phi, n)
        self.update()

    def rotate_around_point(self, phi, n, point, redraw=True):
        self.axis1 = rotate(self.axis1, phi, n)
        self.axis2 = rotate(self.axis2, phi, n)
        self.rays.rotate_around_point(phi, n, point)
        if redraw:
            self.update()

    def translate(self, v, redraw=True):
        self.point += v
        self.rays.translate(v)
        if redraw:
            self.update()

    axis3 = property(lambda self:np.cross(self.axis1,self.axis2))

    def update(self):
        """
        http://www.pygame.org/docs/ref/surfarray.html#pygame.surfarray.blit_array
        https://www.pygame.org/docs/ref/display.html#pygame.display.flip
        """
        self.rays.render(self.gpu_geometry, self.pixels_gpu, self.alpha_depth, keep_last_render=False)
        pixels = self.pixels_gpu.get()
        pygame.surfarray.blit_array(self.screen, pixels.reshape(self.size))

        if self.headless:
            logger.info("saving screen to %s " % self.savepath )
            pygame.image.save(self.screen, self.savepath)
        else:
            self.window.fill(0)
            self.window.blit(self.screen, (0,0))
            pygame.display.flip()

    def process_event_minimal(self, event):
        if event.type == KEYDOWN:
            if event.key == K_ESCAPE:
                self.done = True
                return

    def process_event(self, event):
        if event.type == MOUSEBUTTONDOWN:
            if event.button == 4:
                v = self.scale*self.axis3/10.0
                logger.info("MOUSEBUTTONDOWN 4 %s " % v )
                self.translate(v)

            elif event.button == 5:
                v = -self.scale*self.axis3/10.0
                logger.info("MOUSEBUTTONDOWN 5 %s " % v )
                self.translate(v)

            elif event.button == 1:
                mouse_position = pygame.mouse.get_pos()  # pixel position: top left(0,0) bottom right: size 
                logger.info("clicked mouse_position %s " % str(mouse_position) )
                self.ball.down( mouse_position )
                self.clicked = True

        elif event.type == MOUSEBUTTONUP:
            logger.info("MOUSEBUTTONUP %s " % event.button)
            if event.button == 1:
                self.clicked = False

        elif event.type == MOUSEMOTION and self.clicked:
            mouse_position = pygame.mouse.get_pos()  

            self.ball.drag( mouse_position )
            angle, axis = self.ball.angle_axis()
            logger.info( "ball angle: %s axis: %s " % (angle, axis) )

            #if pygame.key.get_mods() & KMOD_LCTRL:
            #    logger.info("ctrl modifier : rotating around point")
            self.rotate_around_point(angle, axis, self.point)
            #else:
            #    logger.info("no modifier : rotating")
            #    self.rotate(angle, axis)

        elif event.type == KEYDOWN:
            if event.key == K_ESCAPE:
                self.done = True
                return

            elif event.key == K_EQUALS:
                if self.alpha_depth < self.max_alpha_depth:
                    self.alpha_depth += 1
                    logger.info("increase alpha_depth now %s " % ( self.alpha_depth ))
                    self.update()

            elif event.key == K_MINUS:
                if self.alpha_depth > 1:
                    self.alpha_depth -= 1
                    logger.info("decrease alpha_depth now %s " % ( self.alpha_depth ))
                    self.update()

            elif event.key == K_j:
                self.jump_to_solid(self.solid_index) 
                self.update()

            elif event.key in (K_x,K_y,K_z):
                sign = 1.0
                if pygame.key.get_mods() & (KMOD_LSHIFT | KMOD_RSHIFT):
                    sign = -1.0
                if event.key == K_x:
                    axis = self.axis1 
                elif event.key == K_z:
                    axis = self.axis2 
                elif event.key == K_y:
                    axis = self.axis3
                else:
                    assert 0
                pass
                self.translate( axis * self.step * sign )

            elif event.key in (K_s,K_e):
                scale = 2.0
                if pygame.key.get_mods() & (KMOD_LSHIFT | KMOD_RSHIFT):
                    scale = 0.5
                if event.key == K_s:
                    self.step *= scale
                    logger.info("scaling step size by %s now %s " % ( scale, self.step ))
                elif event.key == K_e:
                    self.scale *= scale
                    logger.info("scaling scale size by %s now %s " % ( scale, self.scale ))
                else:
                    assert 0 
                self.update()

            elif event.key == K_i:
                logger.info("point %s " % self.point )
                logger.info("axis1 %s " % self.axis1 )
                logger.info("axis2 %s " % self.axis2 )
                logger.info("axis3 %s " % self.axis3 )
                logger.info("scale %s " % self.scale )
                logger.info("step  %s " % self.step )

        elif event.type == pygame.QUIT:
            self.done = True
            return


    def _run(self, interactive=False):
        """
        Split off the run implementation to allow testing without
        forking/multiprocessing 


        http://www.pygame.org/docs/ref/surface.html

        Headless mode giving::

           ValueError: no standard masks exist for given bitdepth with alpha


        * :google:`pygame SRCALPHA headless` 
        * http://stackoverflow.com/questions/14948711/in-pygame-how-can-i-save-a-transparent-image-headlessly-to-a-file
        * https://www.pygame.org/docs/ref/display.html#pygame.display.set_mode

        """
        pygame.init()
        flags = 0
        depth = 32 # need to specify depth for headless mode when using pygame.SRCALPHA pixel format
        self.window = pygame.display.set_mode(self.size, flags, depth)  # need to specify depth for headless mode
        self.screen = pygame.Surface(self.size, pygame.SRCALPHA)  # SRCALPHA, the pixel format will include a per-pixel alpha
        pygame.display.set_caption('')

        self.init_camera()
        self.init_gpu()
        self.update()

        if self.headless:
            done = True
        else:
            done = False

        self.done = done
        self.clicked = False
        event_handler = self.process_event if interactive else self.process_event_minimal

        while not self.done:
            self.clock.tick(20)
            for event in pygame.event.get():                
                event_handler(event)

        pygame.display.quit()
        self.context.pop()




def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-a","--lookat", default="0,0,0",   help="Lookat position",type=str)
    parser.add_argument("-e","--eye",   default="0,-1,0", help="Eye position",type=str)
    parser.add_argument("-u","--up",   default="-1,0,0", help="Eye position",type=str)
    parser.add_argument("-s","--solid", default=None,     help="Solid",type=str)
    parser.add_argument("-z","--size", default="640,480", help="Pixel size", type=str)
    parser.add_argument("-d","--alpha-max", default=2, help="AlphaMax", type=int)
    parser.add_argument("-f","--focal-length", default=50, help="FocalLength in mm for 35mm film.", type=int)
    parser.add_argument("-r","--radius", default=0.5, help="Arcball radius factir.", type=float)
    parser.add_argument("-G","--headless", action="store_true", help="Headless pygame for debugging.")
    parser.add_argument("-o","--savepath", default="screen.png", help="Path for image saves")

    parser.add_argument("-p","--path", default=os.environ['DAE_NAME'], help="Path of geometry file.",type=str)
    parser.add_argument("-i","--interactive", action="store_true", help="Interative Mode")
    parser.add_argument("-n","--dryrun", action="store_true", help="Argparse checking.")
    parser.add_argument("-k","--cudaprofile", action="store_true", help="CUDA profiling.")
    
    args = parser.parse_args()
    return args, parser


def main():
    args, parser = parse_args()
    print args
    if args.dryrun:
        return

    if args.cudaprofile:
       os.environ['CUDA_PROFILE']="1" 

    geometry = load_geometry_from_string(args.path)

    kwa = {}
    kwa['lookat']=args.lookat
    kwa['eye']=args.eye
    kwa['up']=args.up
    kwa['solid']=args.solid
    kwa['size']=args.size
    kwa['alpha_max']=args.alpha_max
    kwa['focal_length']=args.focal_length
    kwa['radius']=args.radius 
    kwa['headless']=args.headless
    kwa['savepath']=args.savepath

    camera = SimpleCamera(geometry, **kwa)
    print camera.parameter_summary()
    camera._run(interactive=args.interactive)


if __name__ == '__main__':
    main()


