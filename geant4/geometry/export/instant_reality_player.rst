instant reality player
=======================

In app menu choose `View > Statistic > Keyboard Mapping` for some guidance


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
      (it is far too fast, this seems to not work)
B
       toggle fast ray intersect on/off
C
       toggle Back-Face culling
D
       dump the message List to the System Log
E
       switch tp GEOEXAMINE navigation mode
F
       switch tp FREEFLY navigation mode
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

 
   







