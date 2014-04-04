#!/usr/bin/env python
"""

::

    ipython test_loader.py -i

::

    In [4]: geometry.
    geometry.add_solid            geometry.detector_material    geometry.material2_index      geometry.solid_id             geometry.surface_index        
    geometry.bvh                  geometry.flatten              geometry.mesh                 geometry.solid_rotations      geometry.unique_materials     
    geometry.colors               geometry.material1_index      geometry.solid_displacements  geometry.solids               geometry.unique_surfaces      

    In [6]: geometry.mesh
    Out[6]: <chroma.geometry.Mesh at 0x111809550>


::

    In [1]: mesh.get_bounds()
    Out[1]: 
    (array([-20.86420631, -22.49047852,   3.04893279], dtype=float32),
     array([  58.96843338,   77.50952148,  306.86642456], dtype=float32))

    In [2]: print mesh.triangles
    [[   0    1    2]
     [   1    0    3]
     [   3    0    4]
     ..., 
     [8557 8568 8572]
     [8564 8568 8557]
     [8557 8556 8564]]

    In [3]: print mesh.vertices
    [[  45.25120926   -4.07241631  252.98661804]
     [  46.170578     -4.68898106  254.67219543]
     [  45.71078873   -5.45674658  253.29412842]
     ..., 
     [  14.40251446   60.28953552  168.037323  ]
     [  15.06539631   56.1249733   170.93869019]
     [  12.53142452   60.56853485  172.10993958]]



pygame parachute Segmentation Fault
--------------------------------------


::

    In [8]: chroma.view(geometry)
    Fatal Python error: (pygame parachute) Segmentation Fault




"""
import sys, os
import chroma
import chroma.loader

if __name__ == '__main__':
    path = os.path.expandvars("$VIRTUAL_ENV/src/chroma/chroma/models/liberty.stl")
    geometry = chroma.loader.load_geometry_from_string(path)
    print geometry
    mesh = geometry.mesh
    print mesh 


    #chroma.view(mesh)
    chroma.view_nofork(mesh)




