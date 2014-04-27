#!/usr/bin/env python

import logging
log = logging.getLogger(__name__)
import math
import numpy as np
import OpenGL.GL as gl
import OpenGL.GLUT as glut


class DAEText(object):
    fonts = { 
                   "8_BY_13":dict(code=glut.GLUT_BITMAP_8_BY_13,leading=50.), 
                   "9_BY_15":dict(code=glut.GLUT_BITMAP_9_BY_15), 
            "TIMES_ROMAN_10":dict(code=glut.GLUT_BITMAP_TIMES_ROMAN_10), 
            "TIMES_ROMAN_24":dict(code=glut.GLUT_BITMAP_TIMES_ROMAN_24), 
              "HELVETICA_10":dict(code=glut.GLUT_BITMAP_HELVETICA_10,leading=15.),
              "HELVETICA_12":dict(code=glut.GLUT_BITMAP_HELVETICA_12),
             }

    def __init__(self, font="HELVETICA_10"):
        font = self.fonts[font]
        self.font = font['code']
        self.leading = font.get('leading',20.)

    def check(self):
        self( self.fonts.keys() )

    def __call__(self, lines ):
        gl.glDisable( gl.GL_LIGHTING )
        gl.glDisable( gl.GL_DEPTH_TEST )

        gl.glColor3f( 0.1, 0.1, 0.1 )
        for i, line in enumerate(lines):
            ypos = self.leading * (i+1)
            gl.glRasterPos3f( 2.*self.leading , ypos , 0 )
            for c in line:
                glut.glutBitmapCharacter( self.font, ord(c) )
            pass

        gl.glEnable( gl.GL_LIGHTING )
        gl.glEnable( gl.GL_DEPTH_TEST )


if __name__ == '__main__':
    pass

