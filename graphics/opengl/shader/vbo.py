#!/usr/bin/env python
"""
Lightweight versions for debugging
"""
import logging, math, sys
log = logging.getLogger(__name__)
import numpy as np

import OpenGL.GL as gl
import OpenGL.GLU as glu
import OpenGL.GLUT as glut

gvbo = None


class VertexAttribute(object):
    def __init__(self, count, gltype, stride, offset, index, name, normalized=False):
        self.count = count
        self.gltype = gltype
        self.stride = stride
        self.offset = offset
        self.index = index
        self.name = name
        self.normalized = normalized

    def __repr__(self):
        return "%s %s %s %s %s %s" % (self.index, self.name, self.count, self.gltype, self.stride, self.offset )

    def enable(self):
        gl.glVertexAttribPointer( self.index, self.count, self.gltype, self.normalized, self.stride, self.offset )
        gl.glEnableVertexAttribArray( self.index )

class VertexPosition(object):
    def __init__(self, count, gltype, stride, offset):
        self.count = count
        self.gltype = gltype
        self.stride = stride
        self.offset = offset
    def enable(self):
        gl.glVertexPointer(self.count, self.gltype, self.stride, self.offset)
        gl.glEnableClientState(gl.GL_VERTEX_ARRAY)


class VertexColor(object):
    def __init__(self, count, gltype, stride, offset):
        self.count = count
        self.gltype = gltype
        self.stride = stride
        self.offset = offset
    def enable(self):
        gl.glColorPointer(self.count, self.gltype, self.stride, self.offset)
        gl.glEnableClientState(gl.GL_COLOR_ARRAY)



class VertexBufferAttributes(object):
    gltypes = { 'float32': gl.GL_FLOAT,
                'float'  : gl.GL_DOUBLE, 'float64': gl.GL_DOUBLE,
                'int8'   : gl.GL_BYTE,   'uint8'  : gl.GL_UNSIGNED_BYTE,
                'int16'  : gl.GL_SHORT,  'uint16' : gl.GL_UNSIGNED_SHORT,
                'int32'  : gl.GL_INT,    'uint32' : gl.GL_UNSIGNED_INT }

    def __init__(self, vtx):

        dtype = vtx.dtype
        stride = vtx.data.itemsize

        index = 0 
        offset = 0
        attributes = []
        for name in dtype.names:
            if dtype[name].subdtype is not None:
                gtype = str(dtype[name].subdtype[0])
                count = reduce(lambda x,y:x*y, dtype[name].shape)
            else:
                gtype = str(dtype[name])
                count = 1 
            pass
            gltype = self.gltypes[gtype]

            if name == vtx.position:
                attribute = VertexPosition(count,gltype,stride,offset)
            elif name == vtx.color:
                attribute = VertexColor(count,gltype,stride,offset)
            else:
                attribute = VertexAttribute(count,gltype,stride,offset,index, name)
                index += 1
            pass
            attributes.append(attribute)
            offset += dtype[name].itemsize
        pass
        self.attributes = attributes


    def enable(self):
        for att in self.attributes:
            att.enable()

    #def disable(self):
    #    for att in self.attributes:
    #        att.disable()

    def __repr__(self):
        return "\n".join(map(repr,self.attributes))



class VertexBufferObject(object):
    def __init__(self, vtx , target=gl.GL_ARRAY_BUFFER):
        """
        :param vtx: numpy array 
        """
        self.target = target
        self.vtx = vtx
        self.attribs = VertexBufferAttributes( vtx )
        self.init(vtx.data)

    def init(self, vertices):
        self.vertices = vertices
        self.vertices_id = gl.glGenBuffers(1)
        log.info("init buffer %s " % self.vertices_id)

        gl.glBindBuffer( self.target, self.vertices_id )
        gl.glBufferData( self.target, self.vertices, gl.GL_DYNAMIC_DRAW ) 
        gl.glBindBuffer( self.target, 0 )

    def draw(self, mode=gl.GL_LINE_STRIP):
        log.info("draw") 
    
        gl.glBindBuffer( self.target, self.vertices_id )

        self.attribs.enable()

        gl.glDrawArrays( mode, 0, len(self.vertices))

        #self.attribs.disable()
        gl.glBindBuffer( self.target, 0 )

    def directdraw(self, mode=gl.GL_POINTS, vtx_field='position_weight', col_field='ccolor'):
        gl.glBegin(mode)
        for i in range(len(self.vertices)):
            vtx = self.vertices[vtx_field][i]
            col = self.vertices[col_field][i]
            gl.glVertex(*vtx)
            gl.glColor(*col)
        pass
        gl.glEnd()

    def __repr__(self):
        return "\n".join(map(repr,(self.attribs, self.vertices)))



def update(*args):
    glut.glutTimerFunc(33, update, 0)
    glut.glutPostRedisplay()

def display():
    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
    t = glut.glutGet(glut.GLUT_ELAPSED_TIME)

    rot = t % (10 * 1000)
    theta = 2 * 3.141592 * (rot / 10000.0)

    gl.glLoadIdentity()
    glu.gluLookAt(-10*math.sin(theta), -10*math.cos(theta),   0,
                    0,   0,   0,
                    0,   0,   1)

    gvbo.draw()
    #gvbo.directdraw()

    glut.glutSwapBuffers()


def key(*args):
    if args[0] == '\x1b':
        sys.exit(0);

def reshape(width, height):
    aspect = float(width)/float(height) if (height>0) else 1.0
    gl.glViewport(0, 0, width, height)
    gl.glMatrixMode(gl.GL_PROJECTION)
    gl.glLoadIdentity()
    glu.gluPerspective(45.0, aspect, 1.0, 100.0)
    gl.glMatrixMode(gl.GL_MODELVIEW)
    gl.glLoadIdentity()
    glut.glutPostRedisplay()



class VertexData(object):
    dtype = np.dtype([
            ('position_weight'   ,        np.float32, 4 ),
            ('direction_wavelength',      np.float32, 4 ),
            ('ccolor',                    np.float32, 4 ),
          ])

    position = 'position_weight'
    color = 'ccolor'

    def __init__(self, nvert=10):
        self.data = self.make_data(nvert)

    def make_data(self, nvert):
        data = np.zeros(nvert, self.dtype )

        data['position_weight'][:,:3] = np.random.rand(nvert,3) 
        data['ccolor'][:,:3] = np.random.rand(nvert,3) 
 
        data['position_weight'][:,3] = np.ones(nvert)
        data['ccolor'][:,3] = np.ones(nvert)

        return data


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)
    vtx = VertexData()

    glut.glutInit([])
    glut.glutInitDisplayString("rgba>=8 depth>16 double")
    glut.glutInitWindowSize(1280, 720)
    glut.glutCreateWindow("VBO Lite")


    glut.glutDisplayFunc(display)
    glut.glutReshapeFunc(reshape)
    glut.glutKeyboardFunc(key)

    glut.glutTimerFunc(33, update, 0)

    gl.glEnable(gl.GL_DEPTH_TEST)
    gl.glEnable(gl.GL_POINT_SMOOTH)
    gl.glEnable(gl.GL_LINE_SMOOTH)

    gvbo = VertexBufferObject( vtx )
    print gvbo

    glut.glutMainLoop()


