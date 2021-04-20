pyvista-dir(){  echo $HOME/miniconda3/pkgs/pyvista-0.25.3-py_0/site-packages/pyvista ; }
pyvista-cd(){   cd $(pyvista-dir); }
pyvista-vi(){   vi $BASH_SOURCE ; }
pyvista-env(){  elocal- ; }
pyvista-usage(){ cat << EOU


PyVista : VTK for Humans
==========================

.. goal is to make 3D visualization and analysis approachable to
domain-scientists so they can focus on the research questions at hand.


* https://www.pyvista.org
* https://github.com/pyvista
* https://docs.pyvista.org
* https://banesullivan.com/python-blog/000-intro-to-pyvista.html
* https://docs.pyvista.org/examples/index.html


* https://docs.pyvista.org/examples/02-plot/depth-peeling.html#sphx-glr-examples-02-plot-depth-peeling-py

  Depth peeling is a technique to correctly render translucent geometry. 


Others
--------

* https://yt-project.org/docs/dev/visualizing/plots.html


Precise Control of Viewpoint/Camera to match a render ?
----------------------------------------------------------

* manage to get same camera point and focus as the render but the 
  field of view is very different 


* https://vtk.org/doc/nightly/html/classvtkCamera.html

* https://blenderartists.org/t/whats-the-difference-between-orthographic-view-and-isometric-view/1167101/2

* https://github.com/Kitware/VTK/blob/master/Wrapping/Python/README.md




    epsilon:plotting blyth$ pwd
    /Users/blyth/miniconda3/lib/python3.7/site-packages/pyvista/plotting

    epsilon:plotting blyth$ grep camera.Set *.py 
    background_renderer.py:        self.camera.SetFocalPoint(xc, yc, 0.0)
    background_renderer.py:        self.camera.SetPosition(xc, yc, d)
    background_renderer.py:        self.camera.SetParallelScale(0.5 * yd / self._scale)
    plotting.py:        self.camera.SetThickness(path.length)
    renderer.py:            self.camera.SetPosition(scale_point(self.camera, camera_location[0],
    renderer.py:            self.camera.SetFocalPoint(scale_point(self.camera, camera_location[1],
    renderer.py:            self.camera.SetViewUp(camera_location[2])
    renderer.py:        self.camera.SetFocalPoint(scale_point(self.camera, point, invert=False))
    renderer.py:        self.camera.SetPosition(scale_point(self.camera, point, invert=False))
    renderer.py:        self.camera.SetViewUp(vector)
    renderer.py:        self.camera.SetParallelProjection(True)
    renderer.py:        self.camera.SetParallelProjection(False)
    renderer.py:        self.camera.SetModelTransformMatrix(transform.GetMatrix())
    epsilon:plotting blyth$ 



::

    In [3]: pl.camera                                                                                                                                                                                        
    Out[3]: (vtkRenderingOpenGL2Python.vtkOpenGLCamera)0x16948fde0

    In [4]: dir(pl.camera)                                                                                                                                                                                   
    Out[4]: 
    ['AddObserver',
     'ApplyTransform',
     'Azimuth',
     'BreakOnError',
     'ComputeViewPlaneNormal',
     'DebugOff',
     'DebugOn',
     'DeepCopy',
     'Dolly',
     'Elevation',
     'FastDelete',
     'GetAddressAsString',
     'GetCameraLightTransformMatrix',
     'GetClassName',
     'GetClippingRange',
     'GetCommand',
     'GetCompositeProjectionTransformMatrix',
     'GetDebug',
     'GetDirectionOfProjection',
     'GetDistance',
     'GetExplicitProjectionTransformMatrix',
     'GetEyeAngle',
     'GetEyePlaneNormal',
     'GetEyePosition',
     'GetEyeSeparation',
     'GetEyeTransformMatrix',


    In [39]: pl.camera.GetParallelScale()                                                                                                                                                                    
    Out[39]: 1021.7520079765083




    In [30]: m = pl.camera.GetViewTransformMatrix()                                                                                                                                                          
    In [31]: print(str(m))                                                                                                                                                                                   
    vtkMatrix4x4 (0x7fbd48f2cb30)
      Debug: Off
      Modified Time: 27087
      Reference Count: 2
      Registered Events: (none)
      Elements:
        0.707107 -0.707107 0 0 
        0.408248 0.408248 0.816497 0 
        -0.57735 -0.57735 0.57735 -3018.96 
        0 0 0 1 

    In [32]: m = pl.camera.GetEyeTransformMatrix()                                                                                                                                                           

    In [33]: print(str(m))                                                                                                                                                                                   
    vtkMatrix4x4 (0x7fbd48f2bdc0)
      Debug: Off
      Modified Time: 1182
      Reference Count: 2
      Registered Events: (none)
      Elements:
        1 0 0 0 
        0 1 0 0 
        0 0 1 0 
        0 0 0 1 







* https://vtk.org/doc/nightly/html/classvtkCamera.html

* https://vtk.org/doc/nightly/html/classvtkRenderer.html

* https://vtk.org/doc/nightly/html/classvtkRenderer.html#ae8055043e676defbbacff6f1ea65ad1e



    0100 class Renderer(vtkRenderer):
    0101     """Renderer class."""
    0102 
    ....
    1260     def reset_camera(self):
    1261         """Reset the camera of the active render window.
    1262 
    1263         The camera slides along the vector defined from camera position to focal point
    1264         until all of the actors can be seen.
    1265 
    1266         """
    1267         self.ResetCamera()
    1268         self.parent.render()
    1269         self.Modified()
    1270 
    ....
    1279     def view_isometric(self, negative=False):
    1280         """Reset the camera to a default isometric view.
    1281 
    1282         The view will show all the actors in the scene.
    1283 
    1284         """
    1285         self.camera_position = CameraPosition(*self.get_default_cam_pos(negative=negative))
    1286         self.camera_set = False
    1287         return self.reset_camera()


    1198     def set_scale(self, xscale=None, yscale=None, zscale=None, reset_camera=True):
    1199         """Scale all the datasets in the scene.
    1200 
    1201         Scaling in performed independently on the X, Y and Z axis.
    1202         A scale of zero is illegal and will be replaced with one.
    1203 
    1204         """
    1205         if xscale is None:
    1206             xscale = self.scale[0]
    1207         if yscale is None:
    1208             yscale = self.scale[1]
    1209         if zscale is None:
    1210             zscale = self.scale[2]
    1211         self.scale = [xscale, yscale, zscale]
    1212 
    1213         # Update the camera's coordinate system
    1214         transform = vtk.vtkTransform()
    1215         transform.Scale(xscale, yscale, zscale)
    1216         self.camera.SetModelTransformMatrix(transform.GetMatrix())
    1217         self.parent.render()
    1218         if reset_camera:
    1219             self.update_bounds_axes()
    1220             self.reset_camera()
    1221         self.Modified()




::


    In [4]: pl = pv.Plotter()                                                                                                                                                                                

    In [5]: pos = hposi[:,:3]                                                                                                                                                                                

    In [6]: pl.add_points(pos)                                                                                                                                                                               
    Out[6]: (vtkRenderingOpenGL2Python.vtkOpenGLActor)0x17251a9f0


    In [8]: pl.view_vector??                                                                                                                                                                                 
    Signature: pl.view_vector(vector, viewup=None)
    Source:   
        def view_vector(self, vector, viewup=None):
            """Point the camera in the direction of the given vector."""
            focal_pt = self.center
            if viewup is None:
                viewup = rcParams['camera']['viewup']
            cpos = CameraPosition(vector + np.array(focal_pt),
                    focal_pt, viewup)
            self.camera_position = cpos
            return self.reset_camera()
    File:      ~/miniconda3/lib/python3.7/site-packages/pyvista/plotting/renderer.py
    Type:      method




    pl.show?

    cpos : list(tuple(floats))
        The camera position to use

    height : int, optional
        height for panel pane. Only used with panel.

    Return
    ------
    cpos : list
        List of camera position, focal point, and view up



    In [11]: pl.camera_position                                                                                                                                                                              
    Out[11]: 
    [(-20.177947998046875, -20.16827392578125, 3897.8118510121158),
     (-20.177947998046875, -20.16827392578125, 22.314544677734375),
     (0.0, 1.0, 0.0)]

    In [12]: type(pl.camera_position)                                                                                                                                                                        
    Out[12]: pyvista.plotting.renderer.CameraPosition


    In [18]: from pyvista.plotting.renderer import CameraPosition as CP                                                                                                                                      

    In [19]: CP?                                                                                                                                                                                             
    Init signature: CP(position, focal_point, viewup)
    Docstring:      Container to hold camera location attributes.
    Init docstring: Initialize a new camera position descriptor.
    File:           ~/miniconda3/lib/python3.7/site-packages/pyvista/plotting/renderer.py
    Type:           type
    Subclasses:     

    In [20]: CP??                 









Comparisons with other viz tools
---------------------------------

* https://github.com/pyvista/pyvista/issues/146

* https://github.com/marcomusy/vedo

* https://github.com/vispy/vispy

  Direct to OpenGL 


* https://yt-project.org


PyVista vs Mayavi
--------------------

* https://github.com/pyvista/pyvista/issues/146

You're definitely right that there is a lot of overlap in features between
Mayavi and pyvista! I do however think pyvista is approaching 3D viz in a
totally different way the Mayavi... first, pyvista is truly an interface to the
Visualization Toolkit. We provide an easy to use interface to VTK's Python
bindings making accessing VTK data objects simple and fast. This allows pyvista
to merge into any existing Python VTK code as pyvista objects are instances of
VTK objects. It also stays true to VTK's object-oriented approach.

For example, in pyvista we simply wrap common VTK classes with properties and
methods to make accessing the underlying data within the VTK data object and
using VTK filters more intuitive so that users don't need to know the nuances
of creating VTK pipelines and remember all the different VTK classes for
filters, etc. I think Mayavi has a similar effort in this regard but I don't
know enough to comment too much further. I do know that the differences bewteen
how pyvista and Mayavi make VTK filters available to the user are stark:



PyVistaQt
-----------

* https://github.com/pyvista/pyvistaqt
* http://qtdocs.pyvista.org/usage.html

The python package pyvistaqt extends the functionality of pyvista through the
usage of PyQt5. Since PyQt5 operates in a separate thread than VTK, you can
similtaniously have an active VTK plot and a non-blocking Python session.


::

    In [3]: from pyvistaqt import BackgroundPlotter
    In [4]: pl = BackgroundPlotter()
    In [5]: pl.add_mesh(mesh) 


PyVista window interaction 
----------------------------

* https://docs.pyvista.org/plotting/plotting.html#plotting-ref


Model : mesh, cells, nodes, attributes
-----------------------------------------

* https://docs.pyvista.org/getting-started/what-is-a-mesh.html 

Cells aren’t limited to voxels, they could be a triangle between three nodes, a
line between two nodes, or even a single node could be its own cell (but that’s
a special case).

Attributes are data values that live on either the nodes or cells of a mesh


Attributes
~~~~~~~~~~~~

::

    In [75]: mesh.point_arrays                                                                                                                                               
    Out[75]: 
    pyvista DataSetAttributes
    Association: POINT
    Contains keys:
        sample_point_scalars
        VTKorigID

    In [77]: mesh.point_arrays['sample_point_scalars']                                                                                                                       
    Out[77]: 
    pyvista_ndarray([  1,   2,   4,   6,   8,  10,  12,  15,  19,  22,  23,
                      25,  27,  29,  31,  33,  36,  40,  44,  46,  48,  50,
                      52,  54,  56,  58,  60,  63,  65,  67,  69,  71,  73,
                      75,  77,  79,  91,  93,  95,  97,  99, 101, 103, 105,
                     107, 119, 121, 123, 125, 127, 129, 131, 133, 135, 147,
                     149, 151, 153, 155, 157, 159, 161, 163, 175, 177, 179,
                     181, 183, 185, 187, 189, 191, 203, 205, 207, 209, 211,
                     213, 215, 217, 219, 240, 242, 244, 246, 248, 250, 252,
                     254, 256, 286, 288, 290, 292, 294, 296, 298, 300, 302])

    In [79]: mesh.point_arrays['VTKorigID']                                                                                                                                  
    Out[79]: 
    pyvista_ndarray([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13,
                     14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
                     28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
                     42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
                     56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
                     70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83,
                     84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97,
                     98])


    In [76]: mesh.cell_arrays                                                                                                                                                
    Out[76]: 
    pyvista DataSetAttributes
    Association: CELL
    Contains keys:
        sample_cell_scalars


    In [78]: mesh.cell_arrays['sample_cell_scalars']                                                                                                                         
    Out[78]: 
    pyvista_ndarray([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14,
                     15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                     29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40],
                    dtype=int32)

    In [79]:                                    



    In [80]: mesh.point_arrays['my point values'] = np.arange(mesh.n_points)                                                                                                

    In [81]: mesh.plot(scalars='my point values', cpos=bcpos, 
        ...:           show_edges=True, screenshot='beam_point_data.png')                                                                                                   



BackgroundPlotter moved to https://github.com/pyvista/pyvistaqt
------------------------------------------------------------------

::

    In [119]: plotter = pv.BackgroundPlotter()                                                                                                                              
    ---------------------------------------------------------------------------
    QtDeprecationError                        Traceback (most recent call last)
    <ipython-input-119-1a7f685be6b6> in <module>
    ----> 1 plotter = pv.BackgroundPlotter()

    ~/miniconda3/lib/python3.7/site-packages/pyvista/plotting/__init__.py in __init__(self, *args, **kwargs)
         33     def __init__(self, *args, **kwargs):
         34         """Empty init."""
    ---> 35         raise QtDeprecationError('BackgroundPlotter')
         36 
         37 

    QtDeprecationError: `BackgroundPlotter` has moved to pyvistaqt.
        You can install this from PyPI with: `pip install pyvistaqt`
        See https://github.com/pyvista/pyvistaqt






Examples
---------

::

    In [1]: from pyvista import examples 
    In [2]: dir(examples)          



EOU
}


pyvista-gr(){ pyvista-cd ; find . -name '*.py' -exec grep -H "${1:-UnstructuredGrid}" {} \; ; }
pyvista-gl(){ pyvista-cd ; find . -name '*.py' -exec grep -l "${1:-UnstructuredGrid}" {} \; ; }
