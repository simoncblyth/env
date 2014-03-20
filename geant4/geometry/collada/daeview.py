#!/usr/bin/env python
"""
Usage::

   daeview.py -n 4998,4815 -t 4815
   daeview.py -n 4998,4815 -t 4815 -e 3,0,0 -p   
   daeview.py -n 4998,4815:4900 -t 4905   -e 10,10,0

   daeview.py -n 4998,4815 -t 4815 -j 4998  -e 4,0,0

      # animated transition between two nodes


Ideas:

* https://threejsdoc.appspot.com/doc/three.js/src.source/extras/controls/TrackballControls.js.html

* certain viewpoints cause divisions by zero, how to avoid ?

* when eye point is in same direction as up, cannot calc the quat::

           numpy.linalg.linalg.LinAlgError: Eigenvalues did not converge

       * maybe just set up to be eye^look

* caution of optical illusion, you need to see as if you are looking 
  through the volume : otherwise the projection can looks wierd, ones
  preception sometimes flips between these impressions

* filled painting technique does not work from inside objects

* wireframe drawing bogs down when eye enters volumes, primitive painting ?
  this tends to happen with small yfov 
    
* near is critical for controlling orthographic view, when 
  interpolating between volumes of different scales need to 
  interpolate near/yfov too ?  change lens on camera as you get close ?

  the yfov dependence is kinda funny for orthographic, but its there

"""

import logging, sys
log = logging.getLogger(__name__)

import numpy as np
import numpy.core.arrayprint as arrayprint
import contextlib

@contextlib.contextmanager
def printoptions(strip_zeros=True, **kwargs):
    """
    http://stackoverflow.com/questions/2891790/pretty-printing-of-numpy-array
    """
    origcall = arrayprint.FloatFormat.__call__
    def __call__(self, x, strip_zeros=strip_zeros):
        return origcall.__call__(self, x, strip_zeros)
    arrayprint.FloatFormat.__call__ = __call__
    original = np.get_printoptions()
    np.set_printoptions(**kwargs)
    yield 
    np.set_printoptions(**original)
    arrayprint.FloatFormat.__call__ = origcall



import socket
import os, sys, pygame, time, math

from env.graphics.pipeline.transform import qrepr
from env.graphics.pipeline.view_transform import ViewTransform
from env.graphics.pipeline.unit_transform import UnitTransform
from env.graphics.pipeline.perspective_transform import PerspectiveTransform
from env.graphics.pipeline.interpolate_transform import InterpolateTransform, InterpolateViewTransform

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

        if self.transform.view.__class__.__name__ in  ('InterpolateTransform','InterpolateViewTransform'):
            v = self.transform.view
            if v.animate:
                self.transform.view.set('fraction', oscillate( frame, 0., 1., 0.01 ))
                #print "%s %s " % (v.fraction, v.position)
            else:
                print "not animating"
        else:
            pass
            #print "cannot animate this view"


  

class InputHandler(object):
    def __init__(self, controller):
        pass
        self.controller = controller
        self.clicked = False

    def exit(self):
        pygame.quit()
        sys.exit()

    def handle(self, event): 
        #print str(event)

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
        


def make_interpolate_transform( t0, t1 , center ):
    """
    Intractable coordinate offset dependence limits usefulness
    of this approach, it works fine in the viscinity of the 
    world origin, but elsewhere the interpolated rotation 
    means that the model is only visible at start and endpoints.

    Note that offsets in the direction of the axis of rotation 
    cause no problem however and model stays visible all the time.

    Attempts to "hide" the real coordinates by sandwiching between 
    centering to/from translation matrixes somehow fails to help.

    """
    vf = InterpolateTransform( t0, t1 , center=center )
    vf.check_endpoints()
    vf.animate = False
    
    qt, angle, axis = vf.transition_quaternion_angle_axis

    with printoptions(precision=3, suppress=True, strip_zeros=False):
        print vf
        print qrepr(qt)

    vf.setFraction(0.5) 
    return vf


class KeyView(object):
    def __init__(self, eye, look, up, unit, name=""):
        self.eye = eye
        self.look = look
        self.up = up
        self.unit = unit 
        self.name = name

    _eye  = property(lambda self:self.unit(self.eye))
    _look = property(lambda self:self.unit(self.look))
    _up   = property(lambda self:self.unit(self.up,w=0))
    _eye_look_up = property(lambda self:(self._eye, self._look, self._up)) 

    def __repr__(self):
        with printoptions(precision=3, suppress=True, strip_zeros=False):
            return "\n".join([
                    "%s %s " % (self.__class__.__name__, self.name),
                    "p_eye  %s eye  %s " % (self.eye,  self._eye),
                    "p_look %s look %s " % (self.look, self._look),
                    "p_up   %s up   %s " % (self.up,   self._up),
                      ])


def main_check_view_interpolation():
    """
    Animates a change of focus between AD and PMT 
    """
    other_index = 4815   # cylindrical ancestor of the PMT
    focus_index = 4998   # a PMT

    other = Model.dae(other_index, bound=True)
    focus = Model.dae(focus_index, bound=True)

    UF = UnitTransform(focus.get_bounds())
    UO = UnitTransform(other.get_bounds())

    axes = Model.axes(2*UF.extent, center=UF((0,0,0)))
    cube = Model.cube(UF.extent/10., center=UF((0,0,0))) 

    models = [axes,cube,other,focus]

    k0 = KeyView( (0,4,4), (0,0,0), (0,1,0), UF )
    k1 = KeyView( (0,4,4), (0,0,0), (0,1,0), UO )

    v0 = ViewTransform( *k0._eye_look_up )
    v1 = ViewTransform( *k1._eye_look_up )

    vf = InterpolateViewTransform( v0, v1 , 0.)

    orthographic = 0   # 0 for perspective, >0 for orthographic, value also acts as scale for orthographic
    yfov = 50     
    near = v0.distance       
    nx, ny, flip, far = 640, 480, True, near*100.

    perspective = PerspectiveTransform()
    perspective.setView( vf )
    perspective.setCamera( near, far, yfov, nx, ny, flip, orthographic )


    controller = TransformController(perspective)
    handler = InputHandler(controller) 
    viewer = Viewer(models, controller, handler, screensize=perspective.screensize)
    viewer.run()




def parse_args(doc):
    import argparse
    parser = argparse.ArgumentParser(doc)
    parser.add_argument("-n","--nodes", default="3152,3153",   help="DAENode.getall node(s) specifier",type=str)
    parser.add_argument("-a","--look",  default="0,0,0",   help="Lookat position",type=str)
    parser.add_argument("-e","--eye",   default="-2,0,0", help="Eye position",type=str)
    parser.add_argument("-u","--up",   default="0,0,1", help="Eye position",type=str)
    parser.add_argument("-t","--target", default=None,     help="Node index of solid on which to focus",type=str)
    parser.add_argument("-z","--size", default="640,480", help="Pixel size", type=str)
    parser.add_argument("-f","--fov",  default=50., help="Vertical field of view in degrees.", type=float)
    parser.add_argument(     "--near",  default=1., help="Scale factor to apply to near distance, from eye to target node center.", type=float)
    parser.add_argument("-F","--noflip",  dest="flip", action="store_false", default=True, help="Pixel y flip.")
    parser.add_argument("-p","--parallel", action="store_true", help="Parallel projection, aka orthographic." )
    parser.add_argument("-s","--pscale", default=1., help="Parallel projection, scale.", type=float  )
    parser.add_argument(     "--path", default=os.environ['DAE_NAME'], help="Path of geometry file.",type=str)
    parser.add_argument("-i","--interactive", action="store_true", help="Interative Mode")
    parser.add_argument("-l","--loglevel", default="INFO", help="INFO/DEBUG/WARN/..")  
    parser.add_argument("-j","--jump", default=None, help="Animated transition to another node.")  
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.loglevel))
    
    fvec_ = lambda _:map(float, _.split(","))
    ivec_ = lambda _:map(int, _.split(","))

    args.eye = fvec_(args.eye) 
    args.look = fvec_(args.look) 
    args.up = fvec_(args.up) 
    args.size = ivec_(args.size) 

    return args, parser




def find_model( models, target ):
    if target is None:return None
    focus_models = filter(lambda _:str(_.index) == target, models)
    if len(focus_models) == 1:
        focus = focus_models[0]
    else:
        focus = None
    return focus



def main():
    args, parser = parse_args(__doc__)

    models = Model.dae(args.nodes, bound=True, path=args.path)
    focus = models[0]
    focus = find_model(models, args.target)
    jump = find_model(models, args.jump)

    unit = UnitTransform(focus.get_bounds())
    key  = KeyView( args.eye, args.look, args.up, unit )

    view = ViewTransform( *key._eye_look_up )

    if not jump is None:
        junit = UnitTransform(jump.get_bounds())
        jkey  = KeyView( args.eye, args.look, args.up, junit )
        jview = ViewTransform( *jkey._eye_look_up )
        view = InterpolateViewTransform( view, jview , 0.)

    if args.parallel:
        orthographic = args.pscale   
    else:
        orthographic = 0  # perspective

    yfov = args.fov
    near = view.distance*args.near       
    far = near*100.
    nx, ny, flip = args.size[0], args.size[1], args.flip

    perspective = PerspectiveTransform()
    perspective.setView( view )
    perspective.setCamera( near, far, yfov, nx, ny, flip, orthographic )

    controller = TransformController(perspective)
    handler = InputHandler(controller) 
    viewer = Viewer(models, controller, handler, screensize=perspective.screensize)
    viewer.run()


if __name__ == "__main__":
    #main_check_view_interpolation()
    main()

