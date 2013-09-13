instant reality player
=======================


* http://doc.instantreality.org/tutorial/getting-started/
In app menu choose `View > Statistic > Keyboard Mapping` for some guidance



Need to add some viewpoints
----------------------------


From the tutorial::

    Examine Mode Navigation:
    Zoom: Scroll with the mouse wheel or click the right mouse button and move the cursor vertical.
    Shove: Click the middle mouse button and move the cursor around.
    Rotate: Click the left mouse button and move the cursor around or press the arrow keys.

    Cameras: Select a defined Viewpoint in the scene. The Viewpoint node must have the description field set in order to get listed.


Web Interface
---------------

* http://doc.instantreality.org/tutorial/web-interface/

When running the player a web server is available at 

* http://localhost:35668/

This allows to change Node properties, such as `emissiveColor` at URLs like the below.

* http://localhost:35668/Node.html?node=200910384

BUT, the volume names are not currently propagated into VRML, they are just comments by geant4 vrml2file
that I parse and stuff into the shapedb name field. 


Try promoting the comment to actual node name::

    [blyth@belle7 wrl]$ cp around_dupe.wrl around_dupe_names.wrl
    [blyth@belle7 wrl]$ vi around_dupe_names.wrl 
    [blyth@belle7 wrl]$ perl -pi -e 's,#---------- SOLID:,DEF,gc ' around_dupe_names.wrl
    [blyth@belle7 wrl]$ 

Need to fixup the names. As hashes are not appreciated.::

    sqlite> select replace(name,"#","_") from shape limit 10 ;
    replace(name,"#","_")                                                                                                                                                                                   
    ---------------------------------------------------------------------------------------------                                                                                                           
    /dd/Structure/Sites/db-rock.1000                                                                                                                                                                        
    /dd/Geometry/Sites/lvNearSiteRock_pvNearHallTop.1000                                                                                                                                                    
    /dd/Geometry/Sites/lvNearHallTop_pvNearTopCover.1000                                                                                                                                                    
    /dd/Geometry/Sites/lvNearHallTop_pvNearTeleRpc_pvNearTeleRpc:1.1                                                                                                                                        
    /dd/Geometry/RPC/lvRPCMod_pvRPCFoam.1000                                                                                                                                                                
    /dd/Geometry/RPC/lvRPCFoam_pvBarCham14Array_pvBarCham14ArrayOne:1_pvBarCham14Unit.1                                                                                                                     
    /dd/Geometry/RPC/lvRPCBarCham14_pvRPCGasgap14.1000                            


Does VRML2 Support Metadata ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~




Navigation
-----------

Most useful navigation modes keys to swap between:

lookat(l)
          click on volume to go have a look
examine(e)
          3D rotate around by dragging 


Keyboard Mapping
------------------

Find the text of the help message by grepping the dylibs and stringing the hit::

    simon:MacOS blyth$ pwd
    /Users/blyth/Desktop/Instant Player.app/Contents/MacOS
    simon:MacOS blyth$ strings libavalonNavigationNodePool.dylib


+
      increase navigation speed 
-
      decrease navigation speed
      (it is far too fast, this seems to not work, better when you center the coordinates)
B
       toggle fast ray intersect on/off
C
       toggle Back-Face culling
D
       dump the message List to the System Log
E
       switch to GEOEXAMINE navigation mode
F
       switch to FREEFLY navigation mode
G
       grep and dump the current scene to an image file
I
       toggle front collision while navigating
N
       export the backend graph as a BIN file
O 
       switch Occulsion culling mode
R
       Reload the current context trees (e.g. scene)
S
       toggle Small-Feature culling
T
       toggle sorting of transparent objects
V
       export the scene-graph as VRML file
X
       export the scene-graph as X3D file
[
       Decrease the culling feature (e.g. pixel, threshold)
]
       Increase the culling feature (e.g. pixel, threshold)
a
       change camera transformation to show whole scene
       (very useful, to get started)
b
       start the backend web interface
c
       toggle View-Frustum culling   
d
       dump the key mapping to the System Log
e
       switch to EXAMINE navigation mode
f
       switch to FLY navigation mode
g
       switch to GAME navigation mode
h
       toggle head light
i
       toggle lazy Interaction evaluation
l
       switch to LOOKAT navigation mode
       (handy, can click on volumes to go and have a close look)
m
       switch polygon draw mode (point/line/fill)
n
       export the backend graph as ASC file
o
       toggle Occlusion culling
p
       switch to PAN navigation mode
q
       switch to NONE navigation mode
r
       reset view position/orientation to initial values
s
       switch to SLIDE navigation mode
u
       change camera transformation to straighten up
v
       toggle Draw Volume
w
       switch to WALK navigation mode
x
       toggle global Shadow state
{
       switch to prev allowed nav mode
}
       switch to next allowed nav mode
HOME
       switch to the first Viewpoint
END
       switch to the last Viewpoint 
PGUP
       switch to previous Viewpoint
PGDN
       switch to next Viewpoint
UP
       forward navigation command
DOWN
       backward navigation command
LEFT
       left navigation command
RIGHT
       right navigation command
ESC
       escape the immersion, close fullscreen/window
ENTER 
       toggle full screen
SPACE
       switch the info screen foreground

 
   







