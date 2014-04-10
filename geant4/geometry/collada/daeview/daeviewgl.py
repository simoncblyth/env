#!/usr/bin/env python
"""
DAEVIEWGL
==========

.. seealso:: User instructions :doc:`/geant4/geometry/collada/daeview/daeviewgl_usage`


Features
---------

placemarks
~~~~~~~~~~~

The commandline to return to a viewpoint and camera configuration
is written to stdout on exiting or on pressing "W".

solid picking
~~~~~~~~~~~~~~

#. Clicking pixels with mouse/trackpad, yields an (x,y,z) screen position 
   the z value comes from the depth buffer representing the  position of nearest surface.
#. An unprojection transforms the screen space coordinate into world space.
#. This coordinate is then used to determine the list of solids which 
   contain the point within their bounding box. The solid indices, names and 
   extents in mm are written to the screen.
#. The smallest solid is regarded as "picked". Key "O" toggles high-lighting 
   of picked solids with wireframe spheres.

remote control
~~~~~~~~~~~~~~~

A subset of the commandline launch options can be sent over the network to the running 
application. This allows numerical control of viewpoint and camera parameters.::

   udp.py -t 7000 --eye=10,0,0 --look=0,0,0 --up=0,0,1
   udp.py -t 7000_10,0,0_0,0,0_0,0,1                    # equivalent short form

The viewpoint is defined by the `eye` and `look` positions and the `up` direction, which 
are provided in the coordinate frame of the target solid. NB rotations are performed about the 
look position, that is often set to 0,0,0 corresponding to the center of the solid. 
The "K" key toggle markers indicating the eye and look position. 

The options that are accepted interactively are marked with "I" in the options list::

    daeview.sh --help


Small things 
-------------

#. where the current effective "eye" (and maybe "look") points are, 
   just like can touch a solid and see where that is
#. home-ing trackball should home camera too, so return to initial near/far/yfov ? not just position 
#. parallel projection not reflected in the where, move property from scene to camera ?


Issues
--------

near/far wierdness
~~~~~~~~~~~~~~~~~~~~

Changing near can somehow change far clipping. Maybe depth buffer issue.
Seems less prevalent with less extreme near and far.


In Progress
------------

interactive target switching  
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

While pressing "Q" clicking a solid swiches target to that solid.

* currently the launch eye,look,up parameters are adopted, which is jarring 
* adopting the parameters of the prior view is also jarring 
* need to transform transient/offset eye/look/up to world 

* unclear how best to do this : maybe interpolate views 

Intend:

* switch coordinate frame to adopt that of the target, ie switching view
* changes "look" rotation point to the center of the clicked solid
* allows to raycast for any viewpoint without relying on raycasting 
  being fast enough to be interactive 


Next
-----

#. raycast launch control, to avoid GPU panics/system crashes

   * abort subsequent raycast launches when launch times exceed cutoff : 4 seconds
   * add launchdim rather than max_blocks control

#. raycast modifier key to control kernel-flags allowing to 
   interactively switch from image pixels to the
   "time" "tri_count" "node_count" pixels

#. adopt 2D pixel thread blocks, rather than current 1D

   * potential for significant sqrt(32)? speed up, 
     as current 1D block lines of pixels means that many 
     more pixels in a warp of 32 are being held back
   
#. calculate what the trackball translate factor should actually be based on the 
   camera nearsize/farsize, and size of the virtual trackball 
   rather than using adhoc factor

   * will probably need to scale it up anyhow, but would be better not
     to require user tweaking all the time when move between scales

#. take control of lighting, add headlamp (for inside AD)

#. chroma hybrid mode









Ideas
------

#. help text, describing the keys

#. improve screen text flexibility, columns, matrices, ...

#. more use of solid picking, perhaps modal
   
   * present material/surface properties, position in heirarchy 

#. clipping plane controls, that are fixed in world coordinates

#. coloring by material

#. animation speed control

   * speed dependant on distance
   * make changing speed not cause jumps in the interpolation

#. parametric eye movement  



Division of concerns
----------------------

`DAEConfig`
        argument parsing

`DAEGeometry`
        parsing pycollada file and preparing vertex, triangle arrays ready to make VBO

`DAEScene`
        hold state and coordinate

`DAEFrameHandler`
        control of underlying glumpy Frame, presenting graphical view

`DAEInteractivityHandler`
        handle mouse/keyboard inputs and propagate desired actions 

`DAEViewpoint`
         point of view

`DAETrackball`
         rotation and projection, transient offsets from `DAEViewpoint`



"""
import os, logging, socket
log = logging.getLogger(__name__)

import glumpy as gp  

from daeconfig import DAEConfig
from daegeometry import DAEGeometry
from daescene import DAEScene
from daeinteractivityhandler import DAEInteractivityHandler
from daeframehandler import DAEFrameHandler

from env.cuda.cuda_launch import CUDACheck


def main():

    config = DAEConfig(__doc__)
    config.init_parse()
    config.report()
    config.cudacheck = CUDACheck(config)  # MUST be done before CUDA init, for setting of CUDA_PROFILE envvar 

    geometry = DAEGeometry(config.args.nodes, path=config.args.path)
    geometry.flatten()

    figure = gp.Figure(size=config.size)
    frame = figure.add_frame(size=config.frame)

    scene = DAEScene(geometry, config )
    scene.dump() 

    vbo = geometry.make_vbo(scale=scene.scaled_mode, rgba=config.rgba)
    mesh = gp.graphics.VertexBuffer( vbo.data, vbo.faces )

    frame_handler = DAEFrameHandler( frame, mesh, scene )
    fig_handler = DAEInteractivityHandler(figure, frame_handler, scene, config  )


    gp.show()



if __name__ == '__main__':
    main()

