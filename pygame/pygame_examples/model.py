#!/usr/bin/env python

import numpy as np

red = (255,0,0)
cyan = (0,255,255)

green = (0,255,0)
magenta = (255,0,255)

blue = (0,0,255)
yellow = (255,255,0)

white = (255,255,255)
black = (0,0,0)
grey = (127,127,127)


class Model(object):
    """
    """
    @classmethod 
    def cube(cls, halfextent=1. ):
        """ 
        Define the vertices that compose each of the 6 faces. These numbers are
        indices to the vertices list defined above.

        Define colors for each face


             
        +Z plane::

             
             +Y
              |
              |
          4   |   5
              |
              +-------- +X
  
          7       6

             
        -Z plane::

             
             +Y
              |
              |
          0   |   1
              |
              +-------- +X
  
          3       2


        Inconsistent normal directions in vertex order ?
        """
        vertices = halfextent*np.array([
            (-1, 1,-1),
            ( 1, 1,-1),
            ( 1,-1,-1),
            (-1,-1,-1),
            (-1, 1, 1),
            ( 1, 1, 1),
            ( 1,-1, 1),
            (-1,-1, 1)
           ])


        #                      -z        +x        +z        -x         +y       -y          
        groups  = np.array([(0,1,2,3),(1,5,6,2),(5,4,7,6),(4,0,3,7),(0,4,5,1),(3,2,6,7)]) # groupings of vertices, ie quadfaces here
        colors = np.array([yellow,red,blue,cyan,green,magenta])

        gorder = (0,1,2,3,0)   # repeat the first vertex within each primitive to close the quad  
        return cls(vertices, groups, colors, gorder )


    @classmethod 
    def axes(cls, extent):
        vertices = extent*np.array([
            (0,0,0),
            (1,0,0),
            (-1,0,0),
            (0,1,0),
            (0,-1,0),
            (0,0,1),
            (0,0,-1),
        ])
        #                    +x     -x    +y   -y    +z     -z
        groups  = np.array([(0,1),(0,2),(0,3),(0,4),(0,5),(0,6)])
        colors = np.array([red,cyan,green,magenta,blue,yellow])   # +/- rgb/cmy complemetary pairings 
        gorder = (0,1)  # orderings of vertices within the group, allows vertex duplication to close a shape for example 
        return cls(vertices, groups, colors, gorder)

    def __init__(self, vertices, groups, colors, gorder ):
        self.vertices = vertices 
        self.groups = groups 
        self.colors = colors 
        self.gorder = gorder
        print "vertices.shape %s groups.shape %s colors.shape %s " % ( vertices.shape, groups.shape, colors.shape )
        gsize = list(set(map(len,groups)))
        print "gsize %s " % (gsize)
        assert len(gsize) == 1, gsize  # expect consistent groupsize, ie number of vertices in each face
        self.groupsize = gsize[0]

    def primitives(self, transform):
        """
        http://en.wikipedia.org/wiki/Painter%27s_algorithm

        #. collect vertices for each primitive into verts using numpy slice indexing, 
           in the initial world frame coordinates 
        #. transform from world frame into screen frame, the z is still needed for depth ordering

        # pluck first two (x,y) columns from the  (len(points),3 or 4) shaped points  

        """
        points = transform(self.vertices)  # apply all transfomations in one go 

        avgz = []
        for i in range(len(self.groups)):
            group = self.groups[i]
            gpoints = points[group,:]
            avgz.append(np.average(gpoints[:,2]))
        avgz = np.array(avgz)

        for i in avgz.argsort():
            color = self.colors[i]
            group = self.groups[i]
            yield avgz[i], color, points[group,:][self.gorder,:][:,(0,1)]

    def dump_vertices(self, transform):
        print self.vertices
        print transform(self.vertices)

    def dump_primitives(self, transform):
        for avgz, color, points in self.primitives(transform):
            print avgz, color
            print points


if __name__ == '__main__':
   pass
   from env.graphics.pipeline.world_to_screen import PerspectiveTransform

   eye, look, up, near, far = (10,10,10), (0,0,0), (0,1,0), 2, 10
   yfov, nx, ny, flip  = 90, 640, 480, False
      
   transform = PerspectiveTransform()
   transform.setViewpoint( eye, look, up, near, far )
   transform.setCamera( yfov, nx, ny, flip )
 
   #model = Model.cube()
   model = Model.axes(3)

   model.dump_vertices(transform)
   model.dump_primitives(transform)






