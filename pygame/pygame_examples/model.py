#!/usr/bin/env python

import os
import logging
log = logging.getLogger(__name__)
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


rainbow = np.array([red,green,blue,cyan,magenta,yellow,white,black,grey])


def daeload(arg, path=None):
    if path is None:
        path = os.environ['DAE_NAME']
    from env.geant4.geometry.collada.g4daenode import DAENode 
    if len(DAENode.registry) == 0:
        DAENode.parse(path)
    return DAENode.getall(arg)


class Model(object):
    """
    """
    @classmethod
    def dae(cls, arg="3166", bound=True, path=None):
        """
        :param index:
        :param bound: when True provides world frame coordinates, otherwise unplaced local geometry coordinates
        """
        nodes = daeload(arg, path)
        return [cls.dae_one(node, bound=bound) for node in nodes]

    @classmethod
    def dae_one(cls, node, bound=True):
        if bound:
            pl = list(node.boundgeom.primitives())[0]
        else:
            pl = node.geo.geometry.primitives[0]

        tris = pl.triangleset()
        #gorder = (0,1,2,0) # filled tris
        gorder = (0,1,2,1)  # this mistake is rendered as wireframe looking better than the filled tris
        #colors = np.tile( grey, (len(tris),1) )
        colors = rainbow[np.random.randint(len(rainbow),size=len(tris))]
        return cls( tris._vertex , tris._vertex_index, colors, gorder, name="dae solid %s" % node.id, index=node.index ) 


    @classmethod 
    def cube(cls, halfextent=1., center=(0,0,0)):
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

        center = np.array(center)[:3]
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
        return cls(vertices + center, groups, colors, gorder, name="cube")


    @classmethod 
    def axes(cls, extent, center=(0,0,0)):
        vertices = extent*np.array([
            (0,0,0),
            (1,0,0),
            (-1,0,0),
            (0,1,0),
            (0,-1,0),
            (0,0,1),
            (0,0,-1),
        ])
        center = np.array(center)[:3]
        #                    +x     -x    +y   -y    +z     -z
        groups  = np.array([(0,1),(0,2),(0,3),(0,4),(0,5),(0,6)])
        colors = np.array([red,cyan,green,magenta,blue,yellow])   # +/- rgb/cmy complemetary pairings 
        gorder = (0,1)  # orderings of vertices within the group, allows vertex duplication to close a shape for example 
        return cls(vertices + center, groups, colors, gorder, name="axes")

    def __init__(self, vertices, groups, colors, gorder, name="", index=-1):
        self.vertices = vertices 
        self.groups = groups 
        self.colors = colors 
        self.gorder = gorder
        self.name = name
        self.index = index
        gsize = list(set(map(len,groups)))
        assert len(gsize) == 1, gsize  # expect consistent groupsize, ie number of vertices in each face
        self.groupsize = gsize[0]

    def __repr__(self):
        return "%s %s " % ( self.__class__.__name__, self.name )

    def get_bounds(self):
        "Return the lower and upper bounds for the mesh as a tuple."
        return np.min(self.vertices, axis=0), np.max(self.vertices, axis=0)

    def get_center(self):
        bounds = self.get_bounds()
        return np.mean(bounds, axis=0) 

    def debug_offset_vertices(self, offset=(0,0,0), center=False):
        """
        :param offset:
        :param center:

        For debugging frame dependence issues. 

        This allows the model vertices to be offet, and for the 
        model vertices to be centered placing the midpoint of the 
        bounds at the origin.
        """
        offset = np.array(offset)
        if center:
            self.vertices += -self.get_center() 
        self.vertices += offset


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


def check_model_bounds(model):
    """
    The single extent comes from the maximum difference along a world frame axis, 
    so calculating bounds back in parameter space will give at least one
    dimension at extemities -1, 1
    """
    from env.graphics.pipeline.unit_transform import UnitTransform
    bounds = model.get_bounds()
    UT = UnitTransform( bounds )
    #print "unit transform for %s\n%s " % (model, UT ) 

    center = UT((0,0,0))
    assert np.allclose( center[:3], UT.center ), "input parameter frame origin expected to correspond to center of solid in world frame"

    # inverse transform on model vertices gets them into parameter frame
    p_vert = UT(model.vertices, inverse=True)
    #print p_vert

    # all are expected to be inside the cube with extremities -1,-1,-1  1,1,1
    p_lower, p_upper = np.min(p_vert, axis=0), np.max(p_vert, axis=0)
    assert np.allclose(np.min(p_lower), -1 )
    assert np.allclose(np.max(p_upper),  1 )


def check_model_coordinates( model ):
    from env.graphics.pipeline.unit_transform import UnitTransform

    bounds = model.get_bounds()
    UT = UnitTransform( bounds )

    pv = UT(model.vertices, inverse=True)    # world to param
    wv = UT(pv[:,:3])                        # param to world
    assert np.allclose( wv[:,:3], model.vertices ) 

    pbb = ((0,0,0),(1,1,1),(-1,-1,-1))
    wbb = UT(pbb)                          # param to world
    ibb = UT(wbb[:,:3], inverse=True)      # world to param
    assert np.allclose( ibb[:,:3], np.array(pbb) ) 


def test_model_coordinates():
    model = Model.dae()
    check_model_coordinates(model)
    check_model_bounds(model)

def tests():
    test_model_coordinates() 


if __name__ == '__main__':
    pass
    logging.basicConfig(level=logging.INFO)
    tests()


