#!/usr/bin/env python
"""
G4DAEVIEW
==========

.. seealso:: User instructions :doc:`/geant4/geometry/collada/g4daeview/g4daeview_usage`

"""
import os, sys, logging
log = logging.getLogger(__name__)

import IPython
import OpenGL

#OpenGL.FORWARD_COMPATIBLE_ONLY = True    
# surprised this does not cause any errors,  : suspect as its not doing anything 
# glumpy uses lots of fixed pipeline functionality 

import glumpy as gp  

from daeconfig import DAEConfig
from daegeometry import DAEGeometry
from daescene import DAEScene
from daeinteractivityhandler import DAEInteractivityHandler
from daeframehandler import DAEFrameHandler
from daemenu import DAEMenu, DAEMenuGLUT

from env.cuda.cuda_launch import CUDACheck



def main():
    config = DAEConfig(__doc__)
    config.init_parse()
    log.info("************  main ")
    config.report()

    if config.args.geocacheupdate:
        config.wipe_geocache()
  
 
    if config.args.cuda_profile: 
        cudacheck = CUDACheck(config)  # MUST be done before pycuda initialization, for setting of CUDA_PROFILE envvar 
    else:
        cudacheck = None
    config.cudacheck = cudacheck


    rmenu_glut = DAEMenuGLUT()
    rmenu = DAEMenu("rtop", backend=rmenu_glut)
    config.rmenu = rmenu

    #log.info("************  DAEGeometry.get ")
    geometry = DAEGeometry.get(config)
    #log.info("************  DAEGeometry.get DONE ")

    if config.args.ipython:
        g = geometry
        IPython.embed()


    chromacachepath = config.chromacachepath
    chroma_geometry = None 
    if config.args.with_chroma:
        from chroma.detector import Detector
        if config.args.geocache or config.args.geocacheupdate:
            chroma_geometry = Detector.get(chromacachepath)  

            # TODO: get rid of these, sideeffects of the geometry conversion
            # and moving parts liable to get out of sync with whats in the cache
            # scrambling surface/material identity 
            from daechromamaterialmap import DAEChromaMaterialMap
            from daechromasurfacemap import DAEChromaSurfaceMap
            from daechromaprocessmap import DAEChromaProcessMap

            geometry.chroma_material_map = DAEChromaMaterialMap.fromjson(config)
            geometry.chroma_surface_map = DAEChromaSurfaceMap.fromjson(config)
            geometry.chroma_process_map = DAEChromaProcessMap.fromjson(config)
        pass
        if chroma_geometry is None: 
            chroma_geometry = geometry.make_chroma_geometry() 
            if config.args.geocache or config.args.geocacheupdate:
                log.info("as geocache enabled and just created chroma_geometry save to %s " % chromacachepath )
                chroma_geometry.save(chromacachepath)
            pass
        pass

    log.info("************  chroma geometry DONE ")

    figure = gp.Figure(size=config.size)
    frame = figure.add_frame(size=config.frame)
    rmenu_glut.setup_glutMenuStatusFunc()   # probably needs to be after OpenGL context creation

    log.info("************  DAEScene creation ")
    scene = DAEScene(geometry, chroma_geometry, config )

    log.info("************  VBO setup ")
    vbo = geometry.make_vbo(scale=scene.scaled_mode, rgba=config.rgba, index=-1)

    log.info("vbo.data %s " % repr(vbo.data))

    mesh = gp.graphics.VertexBuffer( vbo.data, vbo.faces )

    log.info("************  DAEFrameHandler ")
    frame_handler = DAEFrameHandler( frame, mesh, scene )
    config.glinfo = frame_handler.glinfo()

    log.info("************  DAEInteractivityHandler ")
    fig_handler = DAEInteractivityHandler(figure, frame_handler, scene, config  )
    frame_handler.fig_handler = fig_handler
    scene.fig_handler = fig_handler    # this suggests need to improve architecture


    rmenu.push_handlers(fig_handler)   # so events from rmenu such as on_needs_redraw are routed to the fig_handler
    log.info("************  rmenu_glut.create ")
    rmenu_glut.create(rmenu, "RIGHT")

    log.info("************  deferred apply_launch_config ")
    scene.event.apply_launch_config()

    
    log.info("************  enter eventloop ")
    gp.show()



if __name__ == '__main__':
    main()

