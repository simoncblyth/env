#!/usr/bin/env python
"""
Started from http://codentronix.com/2011/05/12/rotating-3d-cube-using-python-and-pygame/

Changes:

#. split off the model
#. add axes
#. move to quaternion controlled rotation

"""
import numpy as np
import sys, math, pygame
from operator import itemgetter
import socket


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


class Quaternion(object):
    """
    https://github.com/mrdoob/three.js/blob/master/src/math/Quaternion.js
    """
    @classmethod
    def fromAxisAngle(cls, axis, angle, normalize=False):
        q = cls()
        q.setFromAxisAngle(axis, angle, normalize=normalize)
        return q

    def __init__(self, x = 0, y = 0, z = 0, w = None):
        self.x, self.y, self.z = float(x), float(y), float(z)
        self.w = 1. if w is None else w ;

    def setFromAxisAngle(self, axis, angle, normalize=False):
        """
        http://www.euclideanspace.com/maths/geometry/rotations/conversions/angleToQuaternion/index.htm
        axis have to be normalized
        """
        if normalize:
            axis = axis / np.linalg.norm(axis)

        halfAngle = angle / 2.
        s = math.sin(halfAngle)

        self.x = axis[0] * s
        self.y = axis[1] * s
        self.z = axis[2] * s
        self.w = math.cos(halfAngle)

    def __repr__(self):
        return "%s %s  (%s,%s,%s) " % (self.__class__.__name__, self.w, self.x,self.y,self.z )   


class Point3D:
    def __init__(self, x = 0, y = 0, z = 0):
        self.x, self.y, self.z = float(x), float(y), float(z)


    def rotateX(self, angle):
        """ Rotates the point around the X axis by the given angle in radians """
        cosa = math.cos(angle)
        sina = math.sin(angle)
        y = self.y * cosa - self.z * sina
        z = self.y * sina + self.z * cosa
        return Point3D(self.x, y, z)
 
    def rotateY(self, angle):
        """ Rotates the point around the Y axis by the given angle in radians. """
        cosa = math.cos(angle)
        sina = math.sin(angle)
        z = self.z * cosa - self.x * sina
        x = self.z * sina + self.x * cosa
        return Point3D(x, self.y, z)
 
    def rotateZ(self, angle):
        """ Rotates the point around the Z axis by the given angle in radians. """
        cosa = math.cos(angle)
        sina = math.sin(angle)
        x = self.x * cosa - self.y * sina
        y = self.x * sina + self.y * cosa
        return Point3D(x, y, self.z)
 
    def project(self, width, height, fov, viewer_distance):
        """ Transforms this 3D point to 2D using a perspective projection. """
        factor = fov / (viewer_distance + self.z)
        x = self.x * factor + width / 2
        y = -self.y * factor + height / 2
        return Point3D(x, y, self.z)


    def copy(self):
        return Point3D(self.x, self.y, self.z)
 
    def applyQuaternion(self, q ):
        """
        https://github.com/mrdoob/three.js/blob/master/src/math/Vector3.js
        """
        x = self.x;
        y = self.y;
        z = self.z;

        qx = q.x;
        qy = q.y;
        qz = q.z;
        qw = q.w;

        # calculate quat * vector

        ix =  qw * x + qy * z - qz * y;
        iy =  qw * y + qz * x - qx * z;
        iz =  qw * z + qx * y - qy * x;
        iw = -qx * x - qy * y - qz * z;

        # calculate result * inverse quat

        self.x = ix * qw + iw * -qx + iy * -qz - iz * -qy;
        self.y = iy * qw + iw * -qy + iz * -qx - ix * -qz;
        self.z = iz * qw + iw * -qz + ix * -qy - iy * -qx;


class Model(object):
    @classmethod 
    def cube(cls):
        """ 
        Define the vertices that compose each of the 6 faces. These numbers are
        indices to the vertices list defined above.

        Define colors for each face
        """
        vertices = [
            Point3D(-1,1,-1),
            Point3D(1,1,-1),
            Point3D(1,-1,-1),
            Point3D(-1,-1,-1),
            Point3D(-1,1,1),
            Point3D(1,1,1),
            Point3D(1,-1,1),
            Point3D(-1,-1,1)
        ]
        groups  = [(0,1,2,3),(1,5,6,2),(5,4,7,6),(4,0,3,7),(0,4,5,1),(3,2,6,7)]
        colors = [(255,0,255),(255,0,0),(0,255,0),(0,0,255),(0,255,255),(255,255,0)]
        return cls(vertices, groups, colors, proj="project_quadfaces")


    @classmethod 
    def axes(cls):
        vertices = [
            Point3D(0,0,0),
            Point3D(5,0,0),
            Point3D(-5,0,0),
            Point3D(0,5,0),
            Point3D(0,-5,0),
            Point3D(0,0,5),
            Point3D(0,0,-5),
        ]
        groups  = [(0,1),(0,2),(0,3),(0,4),(0,5),(0,6)]
        colors = [(255,0,255),(255,0,0),(0,255,0),(0,0,255),(0,255,255),(255,255,0)]
        return cls(vertices, groups, colors, proj="project_bilines")

    def __init__(self, vertices, groups, colors, proj="project_quadfaces"):
        self.vertices = vertices 
        self.rvertices = vertices
        self.groups = groups 
        self.colors = colors 
        self.proj = proj
        self.projector = getattr(self, proj)

    def qrotate(self, q ):
        verts = []
        for i,v in enumerate(self.vertices):
            r = v.copy()
            r.applyQuaternion(q)
            verts.append(r)
        pass
        self.rvertices = verts

    def rotate(self, angle_x, angle_y, angle_z ):
        """
        #. rotate original vertices, storing into rvertices

        0,0,angle gives a rotating square, so are looking up/down Z
        """
        verts = []
        for i,v in enumerate(self.vertices):
            r = v.rotateX(angle_x).rotateY(angle_y).rotateZ(angle_z)  # each rotate is creating a Point3D
            verts.append(r)
        pass
        self.rvertices = verts

    def project_bilines(self, width, height, fov, viewer_distance):
        """
        """
        t = []
        for r in self.rvertices:
            p = r.project(width, height, fov, viewer_distance)
            t.append(p)
        pret = []
        for group_index, g in enumerate(self.groups):
            pointlist = [(t[g[0]].x, t[g[0]].y), (t[g[1]].x, t[g[1]].y),]
            pret.append([pointlist,self.colors[group_index]])
        return pret

    def project_quadfaces(self, width, height, fov, viewer_distance): 
        """
        #. Calculate the average Z values of each face.
        #. Order faces to provide distant faces before the closer ones.
        """
        t = []
        for r in self.rvertices:
            p = r.project(width, height, fov, viewer_distance)
            t.append(p)

        avg_z = []
        i = 0
        for f in self.groups:
            assert len(f) == 4, "expecing 4 faces in group for"
            z = (t[f[0]].z + t[f[1]].z + t[f[2]].z + t[f[3]].z) / 4.0
            avg_z.append([i,z])
            i = i + 1

        pret = []
        for tmp in sorted(avg_z,key=itemgetter(1),reverse=True):
            group_index = tmp[0]
            f = self.groups[group_index]
            pointlist = [(t[f[0]].x, t[f[0]].y), (t[f[1]].x, t[f[1]].y),
                         (t[f[1]].x, t[f[1]].y), (t[f[2]].x, t[f[2]].y),
                         (t[f[2]].x, t[f[2]].y), (t[f[3]].x, t[f[3]].y),
                         (t[f[3]].x, t[f[3]].y), (t[f[0]].x, t[f[0]].y)]
            pret.append([pointlist,self.colors[group_index]])
        return pret 


 

class Simulation:
    def __init__(self, models, size=(640,480), fov=256, viewer_distance=4, caption="Hello"):
        pygame.init()
        self.screen = pygame.display.set_mode(size)
        pygame.display.set_caption(caption)
        self.width = self.screen.get_width()
        self.height = self.screen.get_height() 
        self.fov = fov
        self.viewer_distance = viewer_distance

        self.models = models 
        self.clock = pygame.time.Clock()
        self.angle = 0
        
    def run(self):
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

            for model in self.models:
                if model.proj == 'project_quadfaces':
                    for pointlist, color in model.projector(self.width, self.height, self.fov, self.viewer_distance):
                        pygame.draw.polygon(self.screen,color,pointlist)
                elif model.proj == 'project_bilines':
                    closed = False
                    thickness = 1 
                    for pointlist, color in model.projector(self.width, self.height, self.fov, self.viewer_distance):
                        pygame.draw.lines(self.screen,color,closed,pointlist,thickness)
                else:
                    assert 0

            self.angle += math.pi/180.
            pygame.display.flip()


if __name__ == "__main__":
    cube = Model.cube()
    axes = Model.axes()
    sim = Simulation([cube,axes])
    sim.run()
