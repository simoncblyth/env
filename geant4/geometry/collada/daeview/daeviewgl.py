#!/usr/bin/env python
"""
DAEVIEWGL
==========

.. seealso:: User instructions :doc:`/geant4/geometry/collada/daeview/daeviewgl_usage`

Issues
--------

#. somehow near can end up getting very small and causing problems that look like far clipping

TODO
----

#. clipping plane controls, that are fixed in world coordinates
#. placemarks
#. coloring by material
#. try running on a lesser machine

Division of concerns
----------------------

`DAEConfig`
        argument parsing

`DAEGeometry`
        parsing pycollada file and preparing vertex, triangle arrays ready to make VBO

`DAEScene`
        coordinate the below

`DAEFrameHandler`
        control of underlying glumpy Frame, presenting graphical view

`DAEInteractivityHandler`
        handle mouse/keyboard inputs and propagate desired actions 



"""
import os, logging, socket
log = logging.getLogger(__name__)

import glumpy as gp  

from daeconfig import DAEConfig
from daegeometry import DAEGeometry
from daescene import DAEScene
from daeinteractivityhandler import DAEInteractivityHandler
from daeframehandler import DAEFrameHandler



def main():

    config = DAEConfig(__doc__)
    config.init_parse()
    print config

    geometry = DAEGeometry(config.args.nodes, path=config.args.path)
    geometry.flatten()

    width, height = map(int,config.args.size.split(","))
    figure = gp.Figure(size=(width,height))
    frame = figure.add_frame(size=map(float,config.args.frame.split(",")))

    scene = DAEScene(geometry, config )
    scene.dump() 

    vbo = geometry.make_vbo(scale=scene.scaled_mode, rgba=map(float,config.args.rgba.split(",")) )
    mesh = gp.graphics.VertexBuffer( vbo.data, vbo.faces )

    frame_handler = DAEFrameHandler( frame, mesh, scene )
    fig_handler = DAEInteractivityHandler(figure, frame_handler, scene, config  )


    gp.show()


if __name__ == '__main__':
    main()

