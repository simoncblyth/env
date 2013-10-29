Meshtool
==========

Meshtool provides PyCollada based utilities such as:

#. viewing collada docs (via Panda3D)
#. dumping info on collada docs


Invokation via bash function
------------------------------

Using system python 2.5.1 to enable panda3d usage::

    simon:~ blyth$ meshtool-
    simon:~ blyth$ t meshtool
    meshtool is a function
    meshtool () 
    { 
        /usr/bin/python -c "from meshtool.__main__ import main ; main() " $*
    }


Start Geometry Web Server
--------------------------

::

    simon:~ blyth$ vnode.py --webserver
    2013-10-29 13:08:49,008 env.graphics.collada.pycollada.vnode INFO     /Users/blyth/env/bin/vnode.py
    2013-10-29 13:08:49,012 env.graphics.collada.pycollada.vnode INFO     VNode.parse pycollada parse /usr/local/env/geant4/geometry/xdae/g4_01.dae    
    ...
    2013-10-29 13:09:02,654 env.graphics.collada.pycollada.vnode INFO     starting webserver 
    http://0.0.0.0:8080/



Grab some geometry and view
-----------------------------

::

    simon:~ blyth$ collada-
    simon:~ blyth$ collada-cd
    simon:collada blyth$ pwd
    /usr/local/env/graphics/collada

    simon:collada blyth$ curl -O http://localhost:8080/subcopy/__dd__Geometry__AD__lvOIL--pvAdPmtArray--pvAdPmtArrayRotated--pvAdPmtRingInCyl..1--pvAdPmtInRing..1--pvAdPmtUnit--pvAdPmt0xa8d92d8.0.dae
    simon:collada blyth$ meshtool --load_collada __dd__Geometry__AD__lvOIL--pvAdPmtArray--pvAdPmtArrayRotated--pvAdPmtRingInCyl..1--pvAdPmtInRing..1--pvAdPmtUnit--pvAdPmt0xa8d92d8.0.dae --viewer
 
    ## `--collada_viewer` shows nothing visible, as have no cameras/lights in the file


To edit the DAE download first::

    simon:collada blyth$ curl -sO http://localhost:8080/subcopy/3199.dae
    simon:collada blyth$ meshtool --load_collada 3199.dae --viewer

Can also view direct from a subcopy URL::

    simon:~ blyth$ meshtool --load_collada http://localhost:8080/subcopy/3199.dae --viewer


Meshtool dumping
------------------

`print_scene`
~~~~~~~~~~~~~~

::

    simon:collada blyth$ meshtool --load_collada 3199.dae --print_scene
    Warning: filter 'sander_simplify' disabled because of ImportError: cannot import name combinations
    <Node transforms=1, children=1>
      <NodeNode node=__dd__Geometry__PMT__lvPmtHemi0xa8d6e90>
        <GeometryNode geometry=pmt-hemi0x88414e8>
        <Node transforms=1, children=1>
          <NodeNode node=__dd__Geometry__PMT__lvPmtHemiVacuum0xa8d6c18>
            <GeometryNode geometry=pmt-hemi-vac0x8840ae8>
            <Node transforms=1, children=1>
              <NodeNode node=__dd__Geometry__PMT__lvPmtHemiCathode0xa8d6a18>
                <GeometryNode geometry=pmt-hemi-cathode0x8840560>
              </Node>
            </Node>
            <Node transforms=1, children=1>
              <NodeNode node=__dd__Geometry__PMT__lvPmtHemiBottom0xa8d6ae8>
                <GeometryNode geometry=pmt-hemi-bot0x8840698>
              </Node>
            </Node>
            <Node transforms=1, children=1>
              <NodeNode node=__dd__Geometry__PMT__lvPmtHemiDynode0xa8d6b80>
                <GeometryNode geometry=pmt-hemi-dynode0x8840708>
              </Node>
            </Node>
          </Node>
        </Node>
      </Node>
    </Node>



`print_info`
~~~~~~~~~~~~~~~~

::

    simon:collada blyth$ meshtool --load_collada 3199.dae --print_info 
    Warning: filter 'sander_simplify' disabled because of ImportError: cannot import name combinations
    Cameras: 0
    Lights: 0
    Materials: 4
       <Material id=__dd__Materials__Pyrex0x8885198 effect=__dd__Materials__Pyrex_fx_0x8885198>
       <Material id=__dd__Materials__Vacuum0x88846a0 effect=__dd__Materials__Vacuum_fx_0x88846a0>
       <Material id=__dd__Materials__Bialkali0x88835d8 effect=__dd__Materials__Bialkali_fx_0x88835d8>
       <Material id=__dd__Materials__OpaqueVacuum0x8883810 effect=__dd__Materials__OpaqueVacuum_fx_0x8883810>
    Effects: 4
       <Effect id=__dd__Materials__Pyrex_fx_0x8885198 type=phong>
       <Effect id=__dd__Materials__Vacuum_fx_0x88846a0 type=phong>
       <Effect id=__dd__Materials__Bialkali_fx_0x88835d8 type=phong>
       <Effect id=__dd__Materials__OpaqueVacuum_fx_0x8883810 type=phong>
    Images: 0
    Geometries: 5
       <Geometry id=pmt-hemi0x88414e8, 1 primitives>
          pmt-hemi0x88414e8-Pos: <FloatSource size=360>
          pmt-hemi0x88414e8-Norm: <FloatSource size=596>
          <Polylist length=596>
       <Geometry id=pmt-hemi-vac0x8840ae8, 1 primitives>
          pmt-hemi-vac0x8840ae8-Pos: <FloatSource size=334>
          pmt-hemi-vac0x8840ae8-Norm: <FloatSource size=496>
          <Polylist length=496>
       <Geometry id=pmt-hemi-cathode0x8840560, 1 primitives>
          pmt-hemi-cathode0x8840560-Pos: <FloatSource size=482>
          pmt-hemi-cathode0x8840560-Norm: <FloatSource size=792>
          <Polylist length=792>
       <Geometry id=pmt-hemi-bot0x8840698, 1 primitives>
          pmt-hemi-bot0x8840698-Pos: <FloatSource size=242>
          pmt-hemi-bot0x8840698-Norm: <FloatSource size=264>
          <Polylist length=264>
       <Geometry id=pmt-hemi-dynode0x8840708, 1 primitives>
          pmt-hemi-dynode0x8840708-Pos: <FloatSource size=50>
          pmt-hemi-dynode0x8840708-Norm: <FloatSource size=72>
          <Polylist length=72>



`print_instances`
~~~~~~~~~~~~~~~~~~~

::

    simon:collada blyth$ meshtool --load_collada 3199.dae --print_instances
    Warning: filter 'sander_simplify' disabled because of ImportError: cannot import name combinations
    <GeometryNode geometry=pmt-hemi0x88414e8>
    <GeometryNode geometry=pmt-hemi-vac0x8840ae8>
    <GeometryNode geometry=pmt-hemi-dynode0x8840708>
    <GeometryNode geometry=pmt-hemi-bot0x8840698>
    <GeometryNode geometry=pmt-hemi-cathode0x8840560>
    Total geometries instantiated in default scene: 5


`print_render_info`
~~~~~~~~~~~~~~~~~~~~~

::

    simon:collada blyth$ curl -sO http://localhost:8080/subcopy/0.dae
    simon:collada blyth$ meshtool --load_collada 0.dae --print_render_info
    Warning: filter 'sander_simplify' disabled because of ImportError: cannot import name combinations
    Total texture RAM required: 0.0 bytes
    Total triangles: 2483650
    Total vertices: 6133544
    Total normals: 6133544
    Total texcoords: 0
    Raw number of draw calls: 12230
    Number of draw calls with instance batching: 249
    Number of draw calls with instance and material batching: 36
    Number of lines: 0


`print_bounds`
~~~~~~~~~~~~~~~~

::

    simon:collada blyth$ meshtool --load_collada 0.dae --print_bounds     
    Warning: filter 'sander_simplify' disabled because of ImportError: cannot import name combinations
    Bounds: <<-2400000, -2400000, -2400000>, <2400000, 2400000, 2400000>>
    Center: <0, 0, 0>
    Point farthest from center: <-2400000, -2400000, -2400000> at distance of 4156922




Meshtool structure
--------------------

Filter pipeline structure, /usr/local/env/graphics/collada/meshtool/meshtool/__main__.py::

     35         order = ['Loading',
     36                  'Printing',
     37                  'Simplification',
     38                  'Optimizations',
     39                  'Visualizations',
     40                  'Meta',
     41                  'Operations',
     42                  'Saving']

    

Meshtool panda3d viewer
------------------------


/usr/local/env/graphics/collada/meshtool/meshtool/filters/panda_filters/viewer.py::


     03 from pandacore import setupPandaApp, getBaseNodePath
     04 from pandacontrols import KeyboardMovement, MouseDrag, MouseScaleZoom, MouseCamera, ButtonUtils
     05 
     06 def runViewer(mesh):
     07     p3dApp = setupPandaApp(mesh)
     08     p3dApp.render.analyze()
     09     KeyboardMovement()
     10     ButtonUtils(getBaseNodePath(p3dApp.render))
     11     MouseDrag(getBaseNodePath(p3dApp.render))
     12     MouseScaleZoom(getBaseNodePath(p3dApp.render))
     13     MouseCamera()
     14     p3dApp.run()


controls
~~~~~~~~

/usr/local/env/graphics/collada/meshtool/meshtool/filters/panda_filters/pandacontrols.py::

     07 class KeyboardMovement(DirectObject):
     08     def __init__(self, scale=1.0):
     09         self.scale = scale
     10         self.adjustSpeed()
     11 
     12         #initial values for movement of the camera
     13         self.cam_pos_x = 0
     14         self.cam_pos_y = 0
     15         self.cam_pos_z = 0
     16         self.cam_h = 0
     17         self.cam_p = 0
     18         self.cam_r = 0
     19 
     20         #Each input modifies x,y,z, h,p,r by an amount
     ..                  camera position x,y,z and camera yaw/pitch/roll
     21         key_modifiers = [('w', [True, 0, 0.5, 0, 0, 0, 0]),   # w/s (arrow_up/arrow_down)   towards/away from object  +y/-y 
     22                          ('s', [True, 0, -0.5, 0, 0, 0, 0]),  # 

     23                          ('a', [True, -0.5, 0, 0, 0, 0, 0]),  # a/d camera left/right -x/+x  
     24                          ('d', [True, 0.5, 0, 0, 0, 0, 0]),   #          perspective change apparent

     25                          ('r', [True, 0, 0, 0.5, 0, 0, 0]),   # 
     26                          ('f', [True, 0, 0, -0.5, 0, 0, 0]),  # 

     27                          ('arrow_up', [True, 0, 0.5, 0, 0, 0, 0]),
     28                          ('arrow_down', [True, 0, -0.5, 0, 0, 0, 0]),

     29                          ('arrow_left', [True, 0, 0, 0, 5, 0, 0]),
     30                          ('arrow_right', [True, 0, 0, 0, -5, 0, 0])]
     ..
     38         self.accept('escape', sys.exit)
     39         self.accept('[', self.speedDown)      # multiple keypresses needed to effect much change
     40         self.accept(']', self.speedUp)


::

     ..              
     ..               w   r
     ..              a s d f
     ..                


Panda3D coordinates
~~~~~~~~~~~~~~~~~~~~~~~~

* https://www.panda3d.org/manual/index.php/Common_State_Changes

::

    myNodePath.setPos(X, Y, Z)
    myNodePath.setHpr(Yaw, Pitch, Roll)


By default in Panda3D, the X axis points to the right, the Y axis is forward,
and Z is up. An object's rotation is usually described using Euler angles
called Heading, Pitch, and Roll (sometimes called Yaw, Pitch, and Roll in other
packages) these specify angle rotations in degrees.




