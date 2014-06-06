#!/usr/bin/env python
"""
A structured version of the glumpy obj-viewer.py example, usage::

    python obj-viewer-structured.py $(glumpy-dir)/demos/triceratops.obj

This gets rid of the decorator event handling straightjacket used in the
example, as that approach encourages use of global objects which
do not scale to larger scripts, or facilitate splitting functionality 
into modules.
"""
import numpy as np
import glumpy as gp
import OpenGL.GL as gl



def make_shader():
    """
    http://www.lighthouse3d.com/tutorials/glsl-tutorial/flatten-shader/
    """
    from glumpy.graphics.shader import Shader

    vertex = r"""

    uniform float time;
    varying float intensity;

    void main()
    {

        vec3 lightDir = normalize(vec3(gl_LightSource[0].position));
        intensity = dot(lightDir,gl_Normal);


        vec4 v = vec4(gl_Vertex);
        v.z = sin(5.0*v.x + time)*0.25;   // bizarre geometry distortion

        vec4 c = vec4(gl_Color);
        c.r = 1.0 ;

        gl_FrontColor = c ; 

        //gl_FrontColor = vec4(0.5, gl_Color.g, gl_Color.b, gl_Color.a);  
        //gl_FrontColor = vec4(0.4,0.4,0.8,1.0);

        gl_Position = gl_ModelViewProjectionMatrix*v;
    }
    """

    fragment = r"""
 
    varying float intensity;

    void main()
    {
       vec4 color;


       if (intensity > 0.95)
            color = vec4(1.0,0.5,0.5,1.0);
        else if (intensity > 0.5)
            color = vec4(0.6,0.3,0.3,1.0);
        else if (intensity > 0.25)
            color = vec4(0.4,0.2,0.2,1.0);
        else
            color = vec4(0.2,0.1,0.1,1.0);

        gl_FragColor = color;
        //gl_FragColor = gl_Color;

    }

    """
    return Shader(vertex, fragment)







def load_obj(filename):
    '''
    Load vertices and faces from a wavefront .obj file and generate normals.
    '''
    data = np.genfromtxt(filename, dtype=[('type', np.character, 1),
                                          ('points', np.float32, 3)])

    # Get vertices and faces
    vertices = data['points'][data['type'] == 'v']
    faces = (data['points'][data['type'] == 'f']-1).astype(np.uint32)

    # Build normals
    T = vertices[faces]
    N = np.cross(T[::,1 ]-T[::,0], T[::,2]-T[::,0])
    L = np.sqrt(N[:,0]**2+N[:,1]**2+N[:,2]**2)
    N /= L[:, np.newaxis]
    normals = np.zeros(vertices.shape)
    normals[faces[:,0]] += N
    normals[faces[:,1]] += N
    normals[faces[:,2]] += N
    L = np.sqrt(normals[:,0]**2+normals[:,1]**2+normals[:,2]**2)
    normals /= L[:, np.newaxis]

    # Scale vertices such that object is contained in [-1:+1,-1:+1,-1:+1]
    vmin, vmax =  vertices.min(), vertices.max()
    vertices = 2*(vertices-vmin)/(vmax-vmin) - 1

    return vertices, normals, faces


class Scene(object):
    def __init__(self, mesh, trackball ):
        self.trackball = trackball
        self.mesh = mesh 
        self.time = 0 


class FigHandler(object):
    def __init__(self, fig, scene ):
        fig.push(self)
        self.fig = fig 
        self.scene = scene 

    def on_init(self):
        pass 
        """
        gl.glLightfv (gl.GL_LIGHT0, gl.GL_DIFFUSE, (1.0, 1.0, 1.0, 1.0))
        gl.glLightfv (gl.GL_LIGHT0, gl.GL_AMBIENT, (0.3, 0.3, 0.3, 1.0))
        gl.glLightfv (gl.GL_LIGHT0, gl.GL_SPECULAR,(0.0, 0.0, 0.0, 0.0))
        gl.glLightfv (gl.GL_LIGHT0, gl.GL_POSITION,(2.0, 2.0, 2.0, 0.0))
        gl.glEnable (gl.GL_LIGHTING)
        gl.glEnable (gl.GL_LIGHT0)
        """

    def on_mouse_drag(self, x,y,dx,dy,button):
        self.scene.trackball.drag_to(x,y,dx,dy)
        self.fig.redraw()

    def on_draw(self):
        self.fig.clear(0.85,0.85,0.85,1)



class FrameHandler(object):
    def __init__(self, frame, scene, shader=None ):
        frame.push(self)
        self.frame = frame
        self.scene = scene
        self.shader = shader
        self.time = 0 

    def on_draw(self):
        self.frame.lock()
        self.frame.draw()
        self.scene.trackball.push()

        self.time += 0.1

        if not self.shader is None:
            self.shader.bind() 
            self.shader.uniformf( 'time',  self.time )

        # faces
        gl.glEnable( gl.GL_POLYGON_OFFSET_FILL )
        gl.glPolygonOffset (1, 1)
        gl.glPolygonMode( gl.GL_FRONT_AND_BACK, gl.GL_FILL )

        self.scene.mesh.draw( gl.GL_TRIANGLES, "pnc" )

        gl.glDisable( gl.GL_POLYGON_OFFSET_FILL )

        # lines
        gl.glPolygonMode( gl.GL_FRONT_AND_BACK, gl.GL_LINE )
        gl.glEnable( gl.GL_BLEND )
        gl.glEnable( gl.GL_LINE_SMOOTH )
        gl.glColor( 0.0, 0.0, 0.0, 0.5 )

        self.scene.mesh.draw( gl.GL_TRIANGLES, "p" )

        gl.glDisable( gl.GL_BLEND )
        gl.glDisable( gl.GL_LINE_SMOOTH )


        if not self.shader is None:
            self.shader.unbind()

        self.scene.trackball.pop()
        self.frame.unlock()



class VertexBufferObject(object):
    def __init__(self, vertices, normals, faces, rgba=np.array([0.5,0.5,0.5,1.]) ):
        nvert = len(vertices)
        data = np.zeros(nvert, [('position', np.float32, 3), 
                                ('color',    np.float32, 4), 
                                ('normal',   np.float32, 3)])
        data['position'] = vertices
        data['color'] = np.tile( rgba, (nvert, 1)) 
        data['normal'] = normals

        self.data = data
        self.faces = faces

    def __repr__(self):
        with printoptions(precision=3, suppress=True, strip_zeros=False):
            return "\n".join([
                   "position",str(self.data['position']),
                   "color",str(self.data['color']),
                   "normal",str(self.data['normal']),
                   "faces",str(self.faces),
                   ])  

    @classmethod
    def from_obj(cls, name):
        vertices, normals, faces = load_obj(name)
        return cls(vertices, normals, faces ) 



if __name__ == '__main__':
    import sys

    vbo = VertexBufferObject.from_obj( sys.argv[1] )

    fig = gp.figure((300,300))
    frame = fig.add_frame()

    mesh = gp.graphics.VertexBuffer( vbo.data, vbo.faces )
    trackball = gp.Trackball( 65, 135, 1.0, 2.5 )

    scene = Scene(mesh, trackball)
    fighandler = FigHandler(fig, scene)
    shader = make_shader()
    framehandler = FrameHandler(frame, scene, shader)

    gp.show()

