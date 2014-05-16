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
from daeviewpoint import DAEViewpoint


class DAETransform(object):
    """
    NB implicit dependance on the scenes active view and corresponding solid 
    """
    def __init__(self, scene, _view=None):
        """
        :param scene:
        :param _view:  defaults to None, meaning that the current active view is accessed implicity via scene.view
                       otherwise when an explicit view instance is provided that will be used 
                       (normally the explicit approach is not appropriate as updates to the 
                       active view will not be honoured)   
        """ 
        self._view = _view
        self.scene = scene       
        self.camera = scene.camera
        self.trackball = scene.trackball
        self.kscale = scene.camera.kscale 

    def _get_view(self):
        return self.scene.view if self._view is None else self._view
    view = property(_get_view)


    def _get_upscale_matrix(self):
        return scale_matrix( self.kscale )    
    upscale = property(_get_upscale_matrix)

    def _get_downscale_matrix(self):
        return scale_matrix( 1./self.kscale )    
    downscale = property(_get_downscale_matrix)


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
                               self.view.world2camera ])
    world2eye = property(_get_world2eye)   # this matches GL_MODELVIEW
  
    def _get_eye2world(self):
        return reduce(np.dot, [self.view.camera2world, 
                               self.view.translate_look2eye, 
                               self.trackball.rotation.T, 
                               self.view.translate_eye2look, 
                               self.trackball.untranslate, 
                               self.upscale])
    eye2world = property(_get_eye2world)

    def _get_eye(self):
        return self.eye2world.dot([0,0,0,1])
    eye = property(_get_eye)


    def _get_nearpoint(self):
        return self.eye2world.dot([0,0,-self.camera.near,1])
    nearpoint = property(_get_nearpoint)

    def _get_gaze(self):
        return self.eye2world.dot([0,0,-1,0])
    gaze = property(_get_gaze)

    def _get_plane(self):
        """
        4 element array containing coefficents of near clipping 
        plane equation (world coordinates)
        """
        normal = self.gaze
        point = self.nearpoint
        eqn = normal 
        eqn[3] = -np.dot(normal, point)
        return eqn 
    plane = property(_get_plane)

    def _get_eye2model(self):
        return reduce(np.dot, [self.view.world2model.matrix,
                               self.view.camera2world, 
                               self.view.translate_look2eye, 
                               self.trackball.rotation.T, 
                               self.view.translate_eye2look, 
                               self.trackball.untranslate, 
                               self.upscale])
    eye2model = property(_get_eye2model)

    def _get_pixel2world_notrackball(self):
        """
        #. it will be getting scaled down so have to scale it up, annoyingly 
        """
        return reduce(np.dot, [self.view.camera2world, 
                               self.upscale, 
                               self.camera.pixel2camera])
    pixel2world_notrackball = property(_get_pixel2world_notrackball)

    def _get_pixel2world(self):
        """ 
        Provides pixel2world matrix that transforms pixel coordinates like (0,0,0,1) or (1023,767,0,1)
        into corresponding world space locations at the near plane for the current camera and view. 
        """
        return reduce(np.dot, [self.view.camera2world, 
                               self.view.translate_look2eye, 
                               self.trackball.rotation.T, 
                               self.view.translate_eye2look, 
                               self.trackball.untranslate, 
                               self.upscale, 
                               self.camera.pixel2camera])
    pixel2world = property(_get_pixel2world)


    def _get_eye_look_up_eye(self):
        """
        #. canonical eye/look/up in the eye frame has a constant "disposition" by definition 
           
           * `eye` at origin
           * `look` along -Z 
           * `up` direction along +Y
           * only the distance between `eye` and `look` can change

        """
        eye_distance = self.view.distance / self.kscale
        return np.vstack([[0,0,0,1], [0,0,-eye_distance,1], [0,eye_distance,0,0]]).T  # eye frame
    eye_look_up_eye = property(_get_eye_look_up_eye) 


    def _get_eye_look_up_model(self):
        """
        :return: model frame coordinates of trackballed eye, look, up  

        Trackball translations do not change the view instance eye however 
        this provides the continuously updating eye/look/up in model frame
        that is the basis for placemarks. 


        Prior approaches to trackball handling that caused confusion
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        #. treating trackball.xyz as an offset
        #. treating trackball.xyz as an absolute position to be transformed
     
        Testing using only the simple special case of translations in the gaze line
        (Z panning backwards) was highly misleading as several incorrect treatments 
        worked for this case but not in general.

        Successful treatment:

        #. consider trackball as a source of translation and rotation transforms
           NOT as providing a coordinate to be transformed 
        #. work with entire MODELVIEW transform sequence at once rather than 
           attempting to operate with partial sequences

        Testing trackball pan conversion to model position
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 
        Standard position to debug from with a wide view 
        and remote command to move view around numerically::

            daeviewgl.py -t 8005 --with-chroma --cuda-profile --near 0.5 --size 640,480
            udp.py --eye=10,10,10
            udp.py --eye=15.5,-6.5,30.2   
            # remote commands change the base view, 
            # so must home the trackball for correspondence with what you see
                 
        #. use remote command to set position `udp.py --eye=10,0,0` (this will home the trackball and change the view)
        #. check the scene "where" position in title bar (after "SC") and eye position (after "e") are the same   
        #. use trackball pan controls (eg spacebar drag down) to move in +Z_eye direction, the SC position should 
           update while the base e position stays fixed, for example ending at SC 20,0,0 and e 10,0,0
        #. issue another remote command, which homes the trackball and sets the view to the 
           SC position `udp.py --eye=20,0,0` there should be no visual jump and the
           base view position  `e 20,0,0` should now match, as have homed


        Final? mystery resolved
        ~~~~~~~~~~~~~~~~~~~~~~~ 

        #. Using (0,0,-self.view.distance,1) for the look point 
           leads to crazy large look positions after trackballing around, 
           as this does not apply the kscale downscale, which it must if the position 
           is in eye frame.

        """
        elu_model = self.eye2model.dot(self.eye_look_up_eye)
        return np.split( elu_model.T.flatten(), 3 )
    eye_look_up_model = property(_get_eye_look_up_model) 

    def _get_eye_look_up_world(self):
        return self.eye2world.dot(self.eye_look_up_eye)
    eye_look_up_world = property(_get_eye_look_up_world) 

    def spawn_view(self):
        eye, look, up = self.eye_look_up_model
        return DAEViewpoint( eye[:3], look[:3], up[:3], self.view.solid, self.view.target )

    def spawn_view_jumping_frame(self, solid ):
        """
        world frame eye point stays unchanged, if trackball is homed as switch to this view
        """
        elu = self.eye_look_up_world
        _elu = solid.world2model.matrix.dot(elu)

        _eye = _elu[:3,0]
        _look = 0,0,0           # set "look" to the center of new solid
        _up  = _elu[:3,2]

        return DAEViewpoint( _eye, _look, _up,  solid, solid.index )


    def __str__(self):
        eye, look, up = self.eye_look_up_model

        s_ = lambda name:"--%(name)s=%(fmt)s" % dict(fmt="%s",name=name) 
        fff_ = lambda name:"--%(name)s=\"%(fmt)s,%(fmt)s,%(fmt)s\"" % dict(fmt="%5.1f",name=name) 

        return   " ".join(map(lambda _:_.replace(" ",""),[
                         s_("target") % self.view.target,
                         fff_("eye")  % tuple(eye[:3]), 
                         fff_("look") % tuple(look[:3]), 
                         fff_("up")   % tuple(up[:3]),
                    #     fff_("norm") % tuple([np.linalg.norm(eye[:3]), np.linalg.norm(look[:3]), np.linalg.norm(up[:3])]),
                          ])) 


    __repr__ = __str__

    def check_modelview(self):
        eye2world = self.eye2world        
        world2eye = self.world2eye        
        check = np.dot( eye2world, world2eye )
        assert np.allclose( check, np.identity(4) ), check   # close, but not close enough in translate column

        if 0:
            print "world2eye\n%s " % world2eye 
            print "eye2world\n%s " % eye2world 
            print "check \n%s" % check




if __name__ == '__main__':
    pass # see test_daetransform.py

 
