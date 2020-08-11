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
