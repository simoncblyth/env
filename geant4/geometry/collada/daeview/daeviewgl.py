#!/usr/bin/env python
"""
DAEVIEWGL
==========

.. seealso:: User instructions :doc:`/geant4/geometry/collada/daeview/daeviewgl_usage`

Issues
--------

near/far wierdness
~~~~~~~~~~~~~~~~~~~~

Changing near can somehow change far clipping. Maybe depth buffer issue.
Seems less prevalent now that have less extreme near and far.

trackball coordinates
~~~~~~~~~~~~~~~~~~~~~~~

Trackball xyz apply offsets to "eye" in eye space, with an
adhoc scaling.
 
* What is the world space coordinate of the offset "eye" ?

  * knowing this would allow creation of "placemarks"

* Which matrix to use exactly  ? 

  * original MODELVIEW of the target "view"
  * continuously updating one as move around 
    (eye is always at origin in eye frame of that one)


Next
-----

#. using CUDA processor to add Chroma simplecamera functionality

#. take control of lighting, add headlamp (for inside AD)

#. calculate what the trackball translate factor should actually be based on the 
   camera nearsize/farsize, rather than using adhoc factor


Ideas
------

#. placemarks, write the commandline to return to a viewpoint and camera configuration
   in response to a key press, this will entail world2model transforms to get the 
   parameters in model frame of the current target volume

#. more use of solid picking
   
   * interactive target switching, so can rotate around what you click 
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

#os.environ['CUDA_PROFILE'] = "1"

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

    cudacheck = CUDACheck(config)

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

    cudacheck.tail()


if __name__ == '__main__':
    main()

