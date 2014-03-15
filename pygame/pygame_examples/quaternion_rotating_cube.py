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


"""
import numpy as np
import socket
import sys, pygame, time, math

from env.graphics.pipeline.world_to_screen import PerspectiveTransform
from env.graphics.transformations.transformations import quaternion_from_matrix, quaternion_matrix, quaternion_about_axis, quaternion_slerp, quaternion_multiply
from env.graphics.transformations.transformations import Arcball

from model import Model 

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
    hmm duplication between matrix and arcball which contains the quaternion is not nice
    some things are easier to set via one and some via the other  
    try split arcball off into a controller
    """
    def __init__(self, transform):
        self.transform = transform
        self.arcball = Arcball(initial=transform.matrix)
        self.clicked = False

    def handle(self, event): 
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.MOUSEBUTTONDOWN:
            print "MOUSEBUTTONDOWN"
            if event.button == 1:
                mouse_position = pygame.mouse.get_pos()  # pixel position: top left(0,0) bottom right: size 
                print "mouse_position ", mouse_position
                self.arcball.down( mouse_position )
                self.clicked = True

        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                print "MOUSEBUTTONUP"
                self.clicked = False

        elif event.type == pygame.MOUSEMOTION and self.clicked:
            print "MOUSEMOTION"
            mouse_position = pygame.mouse.get_pos()  
            self.arcball.drag( mouse_position )
 

        elif event.type == pygame.USEREVENT:
            print "received userevent %s " % event.data
        else:
            pass

    def animate(self, frame):
        """
        """ 
        pass
        #dwiggle = np.array([0,0.1,0.1])*((frame % 10) - 5)
        #self.transform.add( 'look', dwiggle )
        #self.transform.add( 'eye', dwiggle )

        # absolute is easier
        #alook = np.array([0,0,0]) + np.array([0,0.1,0.1])*5*math.cos((frame*math.pi/10.)) 
        #self.transform.set('look', alook )

        #aeye = np.array([5,5,5]) + np.array([0,0.1,0.1])*10*math.cos((frame*math.pi/10.)) 
        #self.transform.set('eye', aeye )

        #lfov,hfov = 5.,90.
        #fov = lfov + (hfov - lfov)/2.*math.cos(frame*math.pi/10.)
        #self.transform.set('yfov', np.clip(fov,5,90))
 
 
class Viewer(object):
    def __init__(self, models, controller, caption="Hello", tick=50., background_color=(150,150,150)):
        pass
        pygame.init()
        screen = pygame.display.set_mode((controller.transform.nx,controller.transform.ny))
        pygame.display.set_caption(caption)
        pass
        self.models = models 
        self.controller = controller
        self.tick = tick 
        self.background_color = background_color
        self.screen = screen 

    def run(self):
        clock = pygame.time.Clock()
        dispatcher = UDPToPygame()
        frame = 0
        while 1:
            for event in pygame.event.get():
                self.controller.handle(event)
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
            if groupsize == 4:
                pygame.draw.polygon(self.screen,color,points)
            elif groupsize == 2:
                closed, thickness = False, 1 
                pygame.draw.lines(self.screen,color,closed,points,thickness)
            else:
                assert 0
        


if __name__ == "__main__":
    models = []
    models.append(Model.cube())
    models.append(Model.axes())

    eye, look, up, near, far = (5,5,5), (0,0,0), (0,0,1), 2, 10
    yfov, nx, ny, flip  = 90, 640, 480, True
      
    transform = PerspectiveTransform()
    transform.setViewpoint( eye, look, up, near, far )
    transform.setCamera( yfov, nx, ny, flip )

    controller = TransformController(transform)
 
    viewer = Viewer(models, controller)
    viewer.run()

    #viewer.draw()
    #time.sleep(2) 
