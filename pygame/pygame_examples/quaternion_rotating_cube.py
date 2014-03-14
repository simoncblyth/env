#!/usr/bin/env python
"""
Started from http://codentronix.com/2011/05/12/rotating-3d-cube-using-python-and-pygame/




Changes:

#. split off the model
#. add axes
#. move to quaternion controlled rotation

Ideas:

* https://threejsdoc.appspot.com/doc/three.js/src.source/extras/controls/TrackballControls.js.html



"""
import numpy as np
import socket
import sys, math, pygame
from three import Point3, Matrix4, Quaternion
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
 

class Simulation:
    def __init__(self, models, size=(640,480), fov=256, viewer_distance=4, caption="Hello"):
        pygame.init()
        self.screen = pygame.display.set_mode(size)
        pygame.display.set_caption(caption)
        self.width = self.screen.get_width()
        self.height = self.screen.get_height() 
        self.aspect = self.width / self.height

        self.fov = fov
        self.viewer_distance = viewer_distance

        self.models = models 
        self.clock = pygame.time.Clock()
        self.angle = 0
        
    def run(self):
        left_right = 10.
        bottom_top = left_right / self.aspect  
        #matrix = Matrix4.fromSymmetricFrustum(left_right, bottom_top, near, far )

        near, far = 1. , 50.
        yfov = 30.    
        matrix = Matrix4.fromPerspective(yfov, self.aspect, near, far )
        fudge = 1.
        screen_ = lambda xy:( (fudge*xy[0]*self.width/2./left_right) + self.width/2. , (fudge*xy[1]*self.height/2./bottom_top) + self.height/2. )

        dispatcher = UDPToPygame()
        while 1:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.USEREVENT:
                    print "received userevent %s " % event.data
                else:
                    pass

            dispatcher.update()
            self.clock.tick(50)
            background_color = (150,150,150) # grey
            self.screen.fill(background_color)

            axis = [0,1,0]
            q = Quaternion.fromAxisAngle(axis, self.angle, normalize=True)
            for model in self.models:
                model.qrotate(q)

            #if 0:
            for model in self.models:
                for pointlist, color in model.project_unrotated(matrix):
                    screenlist = map(screen_, pointlist)
                    print screenlist, color
                    if model.groupsize == 4:
                        pygame.draw.polygon(self.screen,color,screenlist)
                    elif model.groupsize == 2:
                        closed, thickness = False, 1 
                        pygame.draw.lines(self.screen,color,closed,screenlist,thickness)
                    else:
                        assert 0


            #for model in self.models:
            if 0:
                if model.proj == 'project_quadfaces':
                    for pointlist, color in model.projector(self.width, self.height, self.fov, self.viewer_distance):
                        print pointlist
                        pygame.draw.polygon(self.screen,color,pointlist)
                elif model.proj == 'project_bilines':
                    closed, thickness = False, 1 
                    for pointlist, color in model.projector(self.width, self.height, self.fov, self.viewer_distance):
                        pygame.draw.lines(self.screen,color,closed,pointlist,thickness)
                else:
                    assert 0

            #self.angle += math.pi/180.
            pygame.display.flip()


if __name__ == "__main__":
    models = []
    #models.append(Model.cube())
    models.append(Model.axes())
    sim = Simulation(models)
    sim.run()
