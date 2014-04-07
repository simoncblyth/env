#!/usr/bin/env python

import logging
log = logging.getLogger(__name__)
import numpy as np
import OpenGL.GL as gl
import OpenGL.GLUT as glut


class DAEIllustrate(object):
    def __init__(self):
        pass 

    def frustum(self, view, lrbtnf ):
        if not hasattr(view, 'camera2world'):
            log.warn("skipping Frustum for interpolated views")
            return

        eye, look, up = np.split( view.eye_look_up, 3 )
        distance = np.linalg.norm( look - eye )

        c2w = view.camera2world

        eye2 = c2w([0,0,0])[:3]
        if not np.allclose( eye2, eye ):
           log.warn("eye2 %s eye %s " % (str(eye2), str(eye)))

        look2 = c2w([0,0,-distance])[:3]
        if not np.allclose( look2, look ):
           log.warn("look2 %s look %s " % (str(look2), str(look)))

        gl.glDisable( gl.GL_LIGHTING )
        gl.glDisable( gl.GL_DEPTH_TEST )

        gl.glColor3f( 1.,0.,0. )               # red infront at -Z
        self.zdraw( view, lrbtnf, -1 , eye )   # near and far planes are at z=-n and z=-f (n and f positive)  

        gl.glColor3f( 0.,1.,0. )               # green behind at +Z
        self.zdraw( view, lrbtnf, +1 , eye )   
 
        gl.glColor3f( 0.,0.,1. )        # blue eye cube              
        gl.glPushMatrix()
        gl.glTranslate ( *view.eye[:3] )
        glut.glutWireCube( distance/10. )
        gl.glPopMatrix()

        gl.glPushMatrix()
        gl.glTranslate ( *view.look[:3] )
        sc = distance/10.
        glut.glutWireCube( sc )
        #gl.glScalef(sc,sc,sc)
        #glut.glutWireIcosahedron() 
        #glut.glutWireTeapot(sc) 
        gl.glPopMatrix()
 
        gl.glEnable( gl.GL_LIGHTING )
        gl.glEnable( gl.GL_DEPTH_TEST )


    def zdraw(self, view, lrbtnf , zsign, origin ):
        """
        Debug code, speed not an issue

        #. calculate world frame coordinates for the frustrum of a particular view,
           then it will stay fixed as change viewpoint to look back at the frustrum

        """
        pass

        c2w = view.camera2world
        n,f = zsign*lrbtnf[-2:]
        num = 100

        def zquad( lbz, rbz, rtz, ltz, mmz ):
            gl.glBegin(gl.GL_LINE_LOOP)
            gl.glVertex3f( *lbz )
            gl.glVertex3f( *rbz )
            gl.glVertex3f( *rtz )
            gl.glVertex3f( *ltz )
            gl.glEnd()        

        def corners( z ):
            l,r,b,t = lrbtnf[:4] * z / n     # z/n ie n/n -> f/n 
            lbz = c2w([l,b,z])[:3] 
            rbz = c2w([r,b,z])[:3] 
            rtz = c2w([r,t,z])[:3] 
            ltz = c2w([l,t,z])[:3] 
            mmz = c2w([0,0,z])[:3]  # symmetric assumption
            return lbz, rbz, rtz, ltz, mmz 

        for z in np.linspace(n, f, num=num):
            zquad( *corners(z) )        
        pass

        # lines joining near and far corners
        for cn,cf in zip(corners(n), corners(f)):
            gl.glBegin( gl.GL_LINES)
            gl.glVertex3f( *cn )
            gl.glVertex3f( *cf )
            gl.glEnd()
        pass


    def raycast(self, pixel2world, eye, camera ):
        """
        :param pixel2world: matrix represented by 4x4 numpy array 
        :param eye: world frame eye coordinates, typically from view.eye
        :param camera: DAECamera instance, used to get pixel counts, corner pixel coordinates
                       and to provide pixel coordinates for pixel indices 
        """
        corners = np.array(camera.pixel_corners.values())
        wcorners = np.dot( corners, pixel2world.T )           # world frame corners

        #wcorners2 = np.dot( pixel2world, corners.T ).T    pre/post matrix multiplication equivalent when transpose appropriately
        #assert np.allclose( wcorners, wcorners2 )

        indices = np.random.randint(0, camera.npixel,1000)    # random pixel indices 
        pixels = np.array(map(camera.pixel_xyzw, indices ))   # find a numpy way to map, if want to deal with all pixels
        wpoints = np.dot( pixels, pixel2world.T )             # world frame random pixels


        gl.glDisable( gl.GL_LIGHTING )
        gl.glDisable( gl.GL_DEPTH_TEST )
        gl.glColor3f( 0.,0.,1. ) 

        gl.glPointSize(10)
        gl.glBegin( gl.GL_POINTS )
        for wcorner in wcorners:
            gl.glVertex3f( *wcorner[:3] )
            pass 
        gl.glEnd()

        for wcorner in wcorners:
            gl.glBegin( gl.GL_LINES )
            gl.glVertex3f( *eye[:3] )
            gl.glVertex3f( *wcorner[:3] )
            gl.glEnd()
            pass 

        for wpoint in wpoints:
            gl.glBegin( gl.GL_LINES )
            gl.glVertex3f( *eye[:3] )
            gl.glVertex3f( *wpoint[:3] )
            gl.glEnd()
            pass 

        gl.glEnable( gl.GL_LIGHTING )
        gl.glEnable( gl.GL_DEPTH_TEST )



if __name__ == '__main__':
    pass
