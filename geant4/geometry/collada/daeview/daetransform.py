#!/usr/bin/env python
"""
DAETransform
==============

Transforms getting out of hand in daeview, so focus them here 
for greated clarity.


"""

import logging, math
log = logging.getLogger(__name__)
import numpy as np
from daeutil import printoptions, WorldToCamera, CameraToWorld, Transform, translate_matrix, scale_matrix



class DAETransform(object):
    def __init__(self, scene ):
        self.view = scene.view
        self.camera = scene.camera
        self.trackball = scene.trackball
        self.kscale = scene.camera.kscale 

    def _get_upscale_matrix(self):
        return scale_matrix( self.kscale )    
    upscale = property(_get_upscale_matrix)

    def _get_downscale_matrix(self):
        return scale_matrix( 1./self.kscale )    
    downscale = property(_get_downscale_matrix)

    def _get_pixel2world(self):
        """ 
        Provides pixel2world matrix that transforms pixel coordinates like (0,0,0,1) or (1023,767,0,1)
        into corresponding world space locations at the near plane for the current camera and view. 

        Unclear where best to implement this : needs camera, view  and kscale

        #. it will be getting scaled down so have to scale it up, annoyingly 

        TODO: accomodate trackball offsets, so can raycast without being homed on a view
        """
        return reduce(np.dot, [ self.view.camera2world.matrix, self.upscale, self.camera.pixel2camera ])
    pixel2world = property(_get_pixel2world)

    def _get_world2eye(self):
        """
        Objects are transformed from **world** space to **eye** space using GL_MODELVIEW matrix, 
        as daeviewgl regards model spaces as just input parameter conveniences
        that OpenGL never gets to know about those.  

        So need to invert MODELVIEW and apply it to the origin (eye position in eye space)
        to get world position of eye.  Can then convert that into model position.  

        Motivation:

           * determine effective view point (eye,look,up) after trackballing around

        The MODELVIEW sequence of transformations in daeframehandler in OpenGL reverse order, 
        defines exactly what the trackball output means::

            kscale = self.camera.kscale
            distance = view.distance
            gl.glScalef(1./kscale, 1./kscale, 1./kscale)   
            gl.glTranslate ( *trackball.xyz )       # former adhoc 1000. now done internally in trackball.translatefactor
            gl.glTranslate ( 0, 0, -distance )      # shunt back, eye back to origin                   
            gl.glMultMatrixf (trackball._matrix )   # rotation around "look" point
            gl.glTranslate ( 0, 0, +distance )      # look is at (0,0,-distance) in eye frame, so here we shunt to the look
            glu.gluLookAt( *view.eye_look_up )      # NB no scaling, still world distances, eye at origin and point -Z at look

        To get the unproject to dump OpenGL modelview matrix, touch a pixel.


        NB "eye" and "camera" are not synonymous now, using nomenclature

        #. **eye frame** is trackballed and **down scaled** and corresponds to GL_MODELVIEW 
        #. **camera frame** is just from eye, look, up

        #. this means eye frame distance between eye and look needs to be down scaled


        """
        return reduce(np.dot, [self.downscale, 
                               self.trackball.translate, 
                               self.view.translate_look2eye,   # (0,0,-distance)
                               self.trackball.rotation, 
                               self.view.translate_eye2look,   # (0,0,+distance)
                               self.view.world2camera.matrix ])
    world2eye = property(_get_world2eye)   # this matches GL_MODELVIEW
  
    def _get_eye2world(self):
        return reduce(np.dot, [self.view.camera2world.matrix, 
                               self.view.translate_look2eye, 
                               self.trackball.rotation.T, 
                               self.view.translate_eye2look, 
                               self.trackball.untranslate, 
                               self.upscale])
    eye2world = property(_get_eye2world)


    def check_modelview(self):
        eye2world = self.eye2world        
        world2eye = self.world2eye        
        check = np.dot( eye2world, world2eye )
        assert np.allclose( check, np.identity(4) ), check   # close, but not close enough in translate column

        if 0:
            print "world2eye\n%s " % world2eye 
            print "eye2world\n%s " % eye2world 
            print "check \n%s" % check


