#!/usr/bin/env python

import numpy as np

class Model(object):
    """
    """
    @classmethod 
    def cube(cls):
        """ 
        Define the vertices that compose each of the 6 faces. These numbers are
        indices to the vertices list defined above.

        Define colors for each face
        """
        vertices = np.array([
            (-1,1,-1),
            (1,1,-1),
            (1,-1,-1),
            (-1,-1,-1),
            (-1,1,1),
            (1,1,1),
            (1,-1,1),
            (-1,-1,1)
           ])
        groups  = np.array([(0,1,2,3),(1,5,6,2),(5,4,7,6),(4,0,3,7),(0,4,5,1),(3,2,6,7)]) # groupings of vertices, ie quadfaces here
        colors = np.array([(255,0,255),(255,0,0),(0,255,0),(0,0,255),(0,255,255),(255,255,0)])
        gorder = (0,1,1,2,2,3,3,0)  # orderings of vertices within the group 
        return cls(vertices, groups, colors, gorder )


    @classmethod 
    def axes(cls):
        vertices = np.array([
            (0,0,0),
            (5,0,0),
            (-5,0,0),
            (0,5,0),
            (0,-5,0),
            (0,0,5),
            (0,0,-5),
        ])
        groups  = np.array([(0,1),(0,2),(0,3),(0,4),(0,5),(0,6)])
        colors = np.array([(255,0,255),(255,0,0),(0,255,0),(0,0,255),(0,255,255),(255,255,0)])
        gorder = (0,1)  # orderings of vertices within the group 
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

    def groups_(self):
        for group, color in zip(self.groups,self.colors):
            verts = self.vertices[self.gorder,:]      # numpy special slice indexing 
            yield group, color, verts




if __name__ == '__main__':
   pass
   from env.graphics.pipeline.world_to_screen import PerspectiveTransform

   eye, look, up, near, far = (10,0,0), (0,0,0), (0,1,0), 2, 10
   yfov, nx, ny, flip  = 90, 640, 480, True
      
   pt = PerspectiveTransform()
   pt.setViewpoint( eye, look, up, near, far )
   pt.setCamera( yfov, nx, ny, flip )
 
   cube = Model.cube()
   for i, v in enumerate(cube.vertices):
       print i, v, pt(v)

   for group, color, verts in cube.groups_():
       print group, color
       print verts

   axes = Model.axes()
   for i, v in enumerate(axes.vertices):
       print i, v, pt(v)

   for group, color, verts in axes.groups_():
       print group, color
       print verts






