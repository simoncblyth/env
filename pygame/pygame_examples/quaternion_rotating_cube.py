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

import logging
log = logging.getLogger(__name__)

import numpy as np
import socket
import os, sys, pygame, time, math

from env.graphics.pipeline.view_transform import ViewTransform
from env.graphics.pipeline.perspective_transform import PerspectiveTransform
from env.graphics.pipeline.interpolate_transform import InterpolateTransform

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
    """
    def __init__(self, transform ):
        """
        :param transform:  Transform subclass instance
        """
        self.transform = transform
        self.original_transform = transform.copy()
        self._matrix = None

    def _get_matrix(self):
        return self.transform.matrix
    matrix = property(_get_matrix)

    def transform_vertex(self, vert ):
        assert vert.shape == (3,)
        vert = np.append(vert, 1.)
        p = np.dot( self.matrix, vert )
        p /= p[3]
        return p 
 
    def transform_vertices(self, verts ):
        """
        :param verts: numpy 2d array of vertices, for example with shape (1000,3)

        Extended homogenous matrix multiplication yields (x,y,z,w) 
        which corresponds to coordinate (x/w,y/w,z/w)  
        This is a trick to allow to represent the "division by z"  needed 
        for perspective transforms with matrices, which normally cannot 
        represent divisions.
 
        Steps:

        #. add extra column of ones, eg shape (1000,4)
        #. matrix pre-multiply the transpose of that 
        #. divide by last column  (xw,yw,zw,w) -> (xw/w,yw/w,zw/w,1) = (x,y,z,1)  whilst still transposed
        #. return the transposed back matrix 

        To do the last column divide while not transposed could do::

            (verts.T/verts[:,(-1)]).T

        """
        assert verts.shape[-1] == 3 and len(verts.shape) == 2, ("unexpected shape", verts.shape )
        v = np.concatenate( (verts, np.ones((len(verts),1))),axis=1 )  # add 4th column of ones 
        vt = np.dot( self.matrix, v.T )
        vt /= vt[-1]   
        return vt.T
 
    def __call__(self,verts):
        return self.transform_vertices(verts)


    def animate(self, frame):
        """
        """ 
        pass
        def oscillate( frame, low, high, speed ):
            return low + (high-low)*(math.cos(frame*math.pi*speed)+1.)/2.


        # have to base from originals, otherwise get incremental change and it wanders off
        #alook = self.original_perspective.look + self.original_perspective.right*5*math.cos((frame*math.pi/50.)) 
        #self.perspective.set('look', alook )

        # this attemps to move viewpoint perpendicular to original gaze 
        #ope = self.original_perspective.eye
        #opr = self.original_perspective.right
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

        self.transform.view( oscillate( frame, 0., 1., 0.01 ))
  

class InputHandler(object):
    def __init__(self, controller):
        pass
        self.controller = controller
        self.clicked = False

    def handle(self, event): 
        print str(event)

        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
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
        for avgz, color, points in model.primitives(self.controller):
            if groupsize == 4:
                pygame.draw.polygon(self.screen,color,points)
            elif groupsize == 2 or groupsize == 3:
                closed, thickness = False, 1 
                pygame.draw.lines(self.screen,color,closed,points,thickness)
            else:
                assert 0, (groupsize)
        


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    models = []
    #models.append(Model.cube(50))  backing cube
    models.append(Model.cube(1.))
    models.append(Model.axes(3.))
    m = Model.dae(3166)
    models.append(m)





if 0:
    X,Y,Z,O = np.array((1,0,0)),np.array((0,1,0)),np.array((0,0,1)),np.array((0,0,0))

    yfov, nx, ny, flip, near, far = 50, 640, 480, False, 2, 10

    v0 = ViewTransform(eye=(10,10,10),  look=(0,0,0), up=Y)
    v1 = ViewTransform(eye=(10,-10,-10), look=(0,0,0),  up=Y)

    view = InterpolateTransform( v0, v1 , fraction=0 )
    assert np.allclose( view.matrix, v0.matrix ) , (view.matrix, v0.matrix )

    perspective = PerspectiveTransform()
    perspective.setView( view )
    perspective.setCamera( near, far, yfov, nx, ny, flip )

    controller = TransformController(perspective)
    handler = InputHandler(controller)
 
    viewer = Viewer(models, controller, handler, screensize=perspective.screensize)
    viewer.run()

    #viewer.draw()
    #time.sleep(2) 
