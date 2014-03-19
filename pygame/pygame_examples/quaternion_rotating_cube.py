#!/usr/bin/env python
"""
Started from http://codentronix.com/2011/05/12/rotating-3d-cube-using-python-and-pygame/


Changes:

#. split off the model
#. add axes
#. move to quaternion controlled rotation

Ideas:

* https://threejsdoc.appspot.com/doc/three.js/src.source/extras/controls/TrackballControls.js.html


* transformations eg rotations need to be applied to the transform not the models, 
  ie not:: 

      q = Quaternion.fromAxisAngle([0,1,0], self.angle, normalize=True)
      for model in self.models:
          model.qrotate(q)

#. quat interpolation between viewpoints for models 
   close to origin work OK, for world coordinate models
   the interpolation is crazy despite the end points being fine

   * translate model vertices to see if numerical problems
     are the explanation

#. certain viewpoints cause divisions by zero, how to avoid ?

#. when eye point is in same direction as up, cannot calc the quat::

           numpy.linalg.linalg.LinAlgError: Eigenvalues did not converge

       * maybe just set up to be eye^look

#. painting technique does not work from inside objects

#. with cube half side of 2, and near of 1 and eye,look (0,0,2),(0,0,0) the
   orthogonal projection precisely fills the screen  


#. caution of optical illusion, you need to see as if you are looking 
   through the volume : otherwise the projection looks wierd


"""

import logging
log = logging.getLogger(__name__)

import numpy as np
import socket
import os, sys, pygame, time, math

from env.graphics.pipeline.view_transform import ViewTransform
from env.graphics.pipeline.unit_transform import UnitTransform
from env.graphics.pipeline.perspective_transform import PerspectiveTransform
from env.graphics.pipeline.interpolate_transform import InterpolateTransform

from model import Model 
    
X,Y,Z,O = np.array((1,0,0)),np.array((0,1,0)),np.array((0,0,1)),np.array((0,0,0))


class UDPToPygame():
    """
    http://stackoverflow.com/questions/11361973/post-event-pygame
    """
    def __init__(self, port=15006, ip="127.0.0.1"):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(0)
        self.sock.bind((ip,port))

    def update(self):
        try:
            data, addr = self.sock.recvfrom(1024)
            ev = pygame.event.Event(pygame.USEREVENT, {'data': data, 'addr': addr})
            pygame.event.post(ev)
        except socket.error:
            pass   


class TransformController(object):
    """ 
    """
    def __init__(self, transform ):
        """
        :param transform:  Transform subclass instance
        """
        self.transform = transform
        self.original_transform = transform.copy()

    def animate(self, frame):
        """
        """ 
        pass
        def oscillate( frame, low, high, speed ):
            return low + (high-low)*(math.sin(frame*math.pi*speed)+1.)/2.


        # have to base from originals, otherwise get incremental change and it wanders off
        #alook = self.original_perspective.look + self.original_perspective.right*5*math.cos((frame*math.pi/50.)) 
        #self.perspective.set('look', alook )

        # this attemps to move viewpoint perpendicular to original gaze 
        #aeye = oscillate( frame, ope - 5.*opr, ope + 5.*opr , 0.01) 
        #self.perspective.set('eye', aeye )

        #angle = 180.*math.cos(frame*math.pi/180.)
        #self.orientation.set('angle', angle )
        #self.orientation.set('center', np.array([1,0,1]) )
        #self.orientation.set('axis',  np.array([0,1,0]) )

        #self.perspective.set('yfov', oscillate( frame, 5., 90., 0.1 ))  # beating heart cube

        # changing near/far has no effect, near far planes are artifical constructs ?
        # what matters is the distance 
        # but, i am projecting onto the near plane ?
        #
        #near = oscillate( frame, 0.1, 10., 0.1 )
        #far = near*10.
        #print "near, far " , near, far
        #self.perspective.set('near', near )
        #self.perspective.set('far', far )

        if self.transform.view.__class__.__name__ == 'InterpolateTransform':
            self.transform.view.set('fraction', oscillate( frame, 0., 1., 0.01 ))

  

class InputHandler(object):
    def __init__(self, controller):
        pass
        self.controller = controller
        self.clicked = False

    def exit(self):
        pygame.quit()
        sys.exit()

    def handle(self, event): 
        print str(event)

        if event.type == pygame.QUIT:
            self.exit() 
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.exit() 
        elif event.type == pygame.MOUSEBUTTONDOWN:
            print "MOUSEBUTTONDOWN"
            if event.button == 1:
                mouse_position = pygame.mouse.get_pos()  # pixel position: top left(0,0) bottom right: size 
                #print "mouse_position ", mouse_position
                #self.arcball.down( mouse_position )
                #self.clicked = True

        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                print "MOUSEBUTTONUP"
                #self.clicked = False

        elif event.type == pygame.MOUSEMOTION and self.clicked:
            print "MOUSEMOTION"
            #mouse_position = pygame.mouse.get_pos()  
            #self.arcball.drag( mouse_position )
 
        elif event.type == pygame.USEREVENT:
            print "received userevent %s " % event.data
        else:
            pass



class Viewer(object):
    def __init__(self, models, controller, handler, screensize=(640,480), caption="-", tick=50., background_color=(150,150,150)):
        pass
        pygame.init()
        screen = pygame.display.set_mode(screensize)
        pygame.display.set_caption(caption)
        pass
        self.models = models 
        self.controller = controller
        self.handler = handler
        pass
        self.tick = tick 
        self.background_color = background_color
        self.screen = screen 

    def run(self):
        clock = pygame.time.Clock()
        dispatcher = UDPToPygame()
        frame = 0
        while 1:
            for event in pygame.event.get():
                self.handler.handle(event)
            pass
            dispatcher.update()
            clock.tick(self.tick)
            self.controller.animate(frame)
            self.draw()
            frame += 1
        pass

    def draw(self):
        self.screen.fill(self.background_color)
        for model in self.models:
            self.render(model)
        pygame.display.flip()

    def render(self, model):
        groupsize = model.groupsize
        for avgz, color, points in model.primitives(self.controller.transform):
            if groupsize == 4 or groupsize == 3:
                pygame.draw.polygon(self.screen,color,points)
            elif groupsize == 2:
                closed, thickness = False, 1 
                pygame.draw.lines(self.screen,color,closed,points,thickness)
            else:
                assert 0, (groupsize)
        
def bounds(vertices):
    return np.min(vertices, axis=0), np.max(vertices, axis=0)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)


    index = 3158
    index = 4998

    target = Model.dae(index, bound=False)

    # offseting the vertices changes the interpolation, not the endpoint presentation
    # need to hide it from the interpolation 
    #target.vertices += np.array((0,0,3e3))

    unit = UnitTransform(target.get_bounds())
    print "extent %s " % unit.extent
    axes = Model.axes(2*unit.extent)
    cube = Model.cube(unit.extent/2., center=(0,0,0)) 

    models = [axes,target]

    yfov, flip = 50, True
    nx, ny = 640, 480

    orthographic = 0               # perspective when 0 
    #orthographic = 1./unit.extent  # the xy scaling to use

    up = (0,0,1)

    # unit transform converts input parameters into world frame 
    v0 = ViewTransform(eye=(2,2,2),  look=(0,0,0), up=up , unit=unit )
    v1 = ViewTransform(eye=(-4,-4,-4), look=(0,0,0), up=up , unit=unit )
    vf = InterpolateTransform( v0, v1 )
    vf.check_endpoints()
    print vf

    view = vf

    near = v0.distance
    far = v0.distance*100.

    perspective = PerspectiveTransform()
    perspective.setView( view )
    perspective.setCamera( near, far, yfov, nx, ny, flip, orthographic )


    #points = perspective(target.vertices)  # apply all transfomations in one go 
    #print bounds(points) 


#if 0:
    controller = TransformController(perspective)
    handler = InputHandler(controller)
 
    viewer = Viewer(models, controller, handler, screensize=perspective.screensize)
    viewer.run()

