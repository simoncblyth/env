import numpy as np
import numpy.core.arrayprint as arrayprint
import contextlib

@contextlib.contextmanager
def printoptions(strip_zeros=True, **kwargs):
    """
    http://stackoverflow.com/questions/2891790/pretty-printing-of-numpy-array
    """
    origcall = arrayprint.FloatFormat.__call__
    def __call__(self, x, strip_zeros=strip_zeros):
        return origcall.__call__(self, x, strip_zeros)
    arrayprint.FloatFormat.__call__ = __call__
    original = np.get_printoptions()
    np.set_printoptions(**kwargs)
    yield 
    np.set_printoptions(**original)
    arrayprint.FloatFormat.__call__ = origcall


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


