#!/usr/bin/env python
"""
"""

import logging
log = logging.getLogger(__name__)
import numpy as np

from daetransform import DAETransform
from daeutil import printoptions, WorldToCamera, CameraToWorld, Transform, translate_matrix, scale_matrix
from daeviewpoint import DAEViewpoint

from daegeometry import DAEMesh
from daetrackball import DAETrackball


def cube_vertices(center, scale):
    center = np.array(center)
    vertices = []
    square_ = lambda _:[[-1,+1,_],[+1,+1,_],[+1,-1,_],[-1,-1,_]]  # clockwise
    cube = square_(-1)+square_(1)
    for d in cube:
        dd = np.array(d)*scale
        vertices.append(dd)
    pass
    vertices += center
    return vertices
 
def random_vertices(center, scale):
    return (np.random.random((10,3)) - 0.5)*scale + np.array(center)


class DummySolid(DAEMesh):
    """

                Y
                | 
          -1,+1 | +1,+1
        ------------------> X
          -1,-1 | +1,-1 
                | 

    """

    

    def __init__(self, center, scale):
        vertices = cube_vertices(center, scale)
        DAEMesh.__init__(self, vertices, [], [])


class DummyTransform(object):
    matrix = np.identity(4)

class DummyTrackball(object):
    rotation = np.identity(4)
    translate = np.identity(4)
    untranslate = np.identity(4)

class DummyCamera(object):
    kscale = 1.

class DummyView(object):
    solid = DummySolid([100,100,100], 10) 
    world2model = property(lambda self:self.solid.world2model)
    model2world = property(lambda self:self.solid.model2world)
    camera2world = DummyTransform()
    
    translate_look2eye = np.identity(4)
    translate_eye2look = np.identity(4)
    distance = 10.   
    target = "dummy"
 
class DummyScene(object):
    trackball = DummyTrackball()
    camera = DummyCamera()
    view = DummyView()


  
def make_solid_and_view(center, scale, target, _eye, _look):
   solid = DummySolid(center, scale)
   check_solid(solid) 
   view = DAEViewpoint( _eye=_eye, _look=_look, _up=(0,0,1), solid=solid, target=target ) 
   return view 

def check_solid( solid):
    if 0:
        print "solid\n", solid
        print "solid.world2model\n",solid.world2model
        print "solid.model2world\n",solid.model2world
    assert np.allclose( np.identity(4), reduce( np.dot, [solid.world2model.matrix, solid.model2world.matrix]))
    assert np.allclose( np.identity(4), reduce( np.dot, [solid.model2world.matrix, solid.world2model.matrix]))

def check_elu_world( transform, _eye, _look):

    _dist = np.linalg.norm(np.array(_look)-np.array(_eye)) # model frame distance 

    elu = transform.eye_look_up_world
    eye = elu[:3,0]
    look = elu[:3,1]
    up = elu[:3,2]
    dist = np.linalg.norm(look-eye)   # world frame distance
    extent = transform.view.solid.extent 

    assert np.allclose( dist/extent, _dist) , "after extent scaling model and world frame distances should be equal"

    if 0:
        print transform 
        print "elu_model\n",transform.eye_look_up_model
        print "elu_world\n",transform.eye_look_up_world
        print "eye", eye, "look", look, "up", up, "dist", dist, "_dist", _dist, "extent", extent

if __name__ == '__main__':

   logging.basicConfig(level=logging.INFO)
   _eye = (2,0,0)   # (-2,0,0) results in world frame z flip, making interpretation more tricky
   _look = (0,0,0)

   v1 = make_solid_and_view([100,100,100],    10, "solid_1", _eye=_eye, _look=_look)
   v2 = make_solid_and_view([-100,-100,-100], 10, "solid_2", _eye=_eye, _look=_look )

   print "v1.solid \n", v1.solid

   trackball = DAETrackball()
   trackball.xyz = (0,0,0)   # beware default (0,0,3) translation in original glumpy trackball, normally overridden to (0,0,0)

   scene = DummyScene()
   scene.view = v1
   scene.trackball = trackball

   transform = DAETransform( scene ) 
   scene.view.transform = transform

   xyzs = [(0,0,0),(100,100,100),(200,200,200)]

   def dump():
       print "\ntrackball ", trackball 
       print "transform ",transform

       elu_m = np.array(transform.eye_look_up_model)
       elu_w = transform.eye_look_up_world
       elu2_m = v2.solid.world2model.matrix.dot(elu_w)

       print "elu_m \n", elu_m
       print "elu_w.T \n", elu_w.T
       print "elu2_m.T \n", elu2_m.T
       print "scene.view \n", scene.view

   for xyz in xyzs:
       trackball.xyz = xyz   
       dump()
   pass
  

   v2s = transform.spawn_view_jumping_frame( v2.solid )

   scene.view = v2s
   trackball.home()



   print "\n*** AFTER JUMP SOLID ***"
   dump()

   #check_solid(transform.view.solid)
   #check_elu_world(transform, _eye, _look)











