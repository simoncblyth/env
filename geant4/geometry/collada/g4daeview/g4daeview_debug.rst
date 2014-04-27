g4daeview debugging
====================

Often loose geometry, ie after making some changes 
find that only get blank screens.  

Tricks to work out whats gone wrong
------------------------------------

#. switch to line mode with `--golight --line`
#. vary near, far, yfov
#. check scaled mode

    g4daeview.py -t +0 --eye "2,2,2"


Issue
------

Near behaving like far
-----------------------

To see the lower parts of the geometry with::

   g4daeview.py -g 3153:12230 -t +5000 --eye=0,0,10 

Have to increase near to 1.01 from initial 0.1.  Why is near doing that ?



Removing the scale kludge
---------------------------

::

     0994  g4daeview.py -g 3153:12230 -t +0
     0995  g4daeview.py -g 3153:12230 -t +0
     0996  g4daeview.py -g 3153:12230 -t +0 --scale=8094.
     0997  g4daeview.py -g 3153:12230 -t +0 --scale=4094.
     0998  g4daeview.py -g 3153:12230 -t +0 --scale=1094.
     0999  g4daeview.py -g 3153:12230 -t +0 --scale=1094.
     1000  g4daeview.py -g 3153:12230 -t +0 --scale=1.
     1001  g4daeview.py -g 3153:12230 -t +0 --scale=100.
     1002  g4daeview.py -g 3153:12230 -t +0 --scale=1.
     1003  g4daeview.py -g 3153:12230 -t +0 --scale=10.
     1004  g4daeview.py -g 3153:12230 -t +0 --scale=10. --flight=10
     1005  g4daeview.py -g 3153:12230 -t +0 --scale=10. --flight=10 --near=10

::

     1043  g4daeview.py -g 3153:12230 -t +5000 --scale=100. --near=30 --flight=20 
     1044  g4daeview.py -g 3153:12230 -t +5000 --scale=100. --near=30 --flight=20 
     1045  g4daeview.py -g 3153:12230 -t +5000 -j 9140_-2,2,2 --scale=100. --near=30 --flight=20 
     1046  g4daeview.py -g 3153:12230 -t +5000 -j 9140_-2,2,2 --scale=100. --near=30 --flight=20 
     1047  g4daeview.py -g 3153:12230 -t +5000 -j 9140_-2,2,2 --scale=100. --near=30 --flight=20 
     1048  g4daeview.py -g 3153:12230 -t +5000 -j 9140_-2,2,2 --scale=100. --near=30 --flight=20 
     1049  g4daeview.py -g 3153:12230 -t +5000 -j 9140_-2,2,2 --scale=100. --near=30 --flight=20 




Targetting outer volumes draw blanks
--------------------------------------

::

    g4daeview.py -g 3153:12230 -t "" --nolight --line     #  blank
    g4daeview.py -g 3153:12230 -t +0 --nolight --line     # 
    g4daeview.py -g 3153:12230 -t +6 --nolight --line     #  +0 to +6 all draw blanks
    g4daeview.py -g 3153:12230 -t 3159 --nolight --line   # absolute equivalent matches

    g4daeview.py -g 3153:12230 -t +7 --nolight --line     #  first visible volume 
    g4daeview.py -g 3153:12230 -t 3160 --nolight --line      

    g4daeview.py -g 3153:12230 -t -10                     # last visible
    g4daeview.py -g 3153:12230 -t -9                      # same issue at end of volume list 
    g4daeview.py -g 3153:12230 -t -1       

    g4daeview.py -g 3153:12230 -t -9 --wlight 0           # lights at infinity dont help 


#. maybe lights contained inside some geometry, so cannot see anything when go outside that volume ? 

   * but light is off ?

#. wierd, via remote control can succeed to see the outer volumes when 

   * startup in situation where see something
   * seems startup is setting the light positions ? which infulence even `--nolight` ?
   * the issue is of handling scale changes, need better feedback into title bar 


Near
-----

Need crazy small near in many situations, why? a scaling problem still::

    g4daeview.py -g 3153:12230 -t "+0" --nearclip 1e-6,1e6
    g4daeview.py -g 3153:12230 -t "" --nearclip 1e-6,1e6


A reasonable near of 10mm gives a blank screen::

    g4daeview.py -g 3153:12230 -t "" --nearclip 1e-6,1e6 --near 10

What near is needed for geometry to show up::

    g4daeview.py -g 3153:12230 -t "" --nearclip 1e-6,1e6 --near .00029        # extent 8094.19   8 meters
    g4daeview.py -g 3153:12230 -t 5000 --nearclip 1e-6,1e6 --near 0.46       # extent  98.14  


Removed some scaling, now a more reasonble near of 0.1mm can show large scale and small scale::

    g4daeview.py -g 3153:12230 -t "" --nearclip 1e-6,1e6 --near 0.1


Currently working::

   g4daeview.py -g 3153:12230 -t "" --nearclip 1e-6,1e6 --near 0.2 --flight 100


Light control
-------------

Moving scaling into MODELVIEW enables light control::

    g4daeview.py -g 3153:12230 -t "" --nearclip 1e-6,1e6 --near 0.2 --flight 1e-6 --eye 2,2,0


Unproject
----------

* http://www.songho.ca/opengl/gl_projectionmatrix.html

After wrapping the unproject calls in matrix push/pop get some coordinates
that look like world ones::

    ouse button pressed (x=789.0, y=550.0, button=2)
    INFO:env.geant4.geometry.collada.daeview.daeframehandler:unproject 789.0 550.0 
    u0 (-16636.554198999842, -796072.4374958207, -1611.3298911357165) 
    u1 (-184837.13407452128, -1098345.946201303, -2228586.9131478006) 
    Mouse button pressed (x=1402.0, y=38.0, button=2)
    INFO:env.geant4.geometry.collada.daeview.daeframehandler:unproject 1402.0 38.0 
    u0 (-16644.035864074554, -796066.1885356996, -1611.3298911357165) 
    u1 (-1679148.3181099917, 149757.3709506284, -2228586.9131478006) 
    Mouse button pressed (x=41.0, y=61.0, button=2)
    INFO:env.geant4.geometry.collada.daeview.daeframehandler:unproject 41.0 61.0 
    u0 (-16627.424858908667, -796066.4692507051, -1611.3298911357165) 
    u1 (1638563.7886964362, 93690.22974414061, -2228586.9131478006) 
    Mouse button pressed (x=803.0, y=528.0, button=2)
    INFO:env.geant4.geometry.collada.daeview.daeframehandler:unproject 803.0 528.0 
    u0 (-16636.725069001546, -796072.1689858156, -1611.3298911357165) 
    u1 (-218964.95883213126, -1044716.5067924673, -2228586.9131478006) 





Trackball
------------

* rotating about eyepoint rather than about a point somewhat ahead of you 


Remote Control
---------------

Targetting different volumes or all volumes is a quick way 
to switch between scenarios::

    udp.py "-t +500"
    udp.py "-t +600"
    udp.py "-t +3"
    udp.py "-t +2"
    udp.py "-t +1"
    udp.py "-t +0"     # first volume
    udp.py "-t """     # all volumes
    udp.py "-t -1"     # last volume
    udp.py "-t -10"


Modes of operation
------------------

scaled mode
~~~~~~~~~~~~~

::

    g4daeview.py


target mode
~~~~~~~~~~~~

::

    g4daeview.py -t "" 
    g4daeview.py -t "" --nolight --line

    g4daeview.py -t +0
    g4daeview.py -t +0    --nolight --line

    g4daeview.py -t +1000 --nolight --line   
 

Scenarios
----------

detail view
~~~~~~~~~~~~

View targetting a single small piece of geometry (eg a PMT) within context 
of many other such pieces of geometry all within containing geometry (eg the AD) 

::

    g4daeview.py -g 3153:6000 -t 5000
    g4daeview.py -g 3153:6000 -t 5000 --nolight --line      # linemode

outside view
~~~~~~~~~~~~~~

::

    g4daeview.py -g 3153:6000 -t ""
    g4daeview.py -g 3153:6000 -t 3153 --nolight --line  




Lighting
----------


In glumpy original scaled geometry mode the lighting is acceptable with false directional RGB and white background::

    g4daeview.py -g 5000:6000  
    g4daeview.py -g 5000:6000 --lights="rgb" --rlight="-1,1,1" --glight="1,1,1" --blight="0,-1,1" --xyz=0,0,3   # the original defaults, giving same appearance

    g4daeview.py -g 5000:6000 --lights="rgb" --rlight="-1,1,-5" --glight="1,1,-5" --blight="0,-1,-5" --xyz=0,0,-3  # white screen, due to position

    g4daeview.py -g 5000:6000 --lights="rgb" --rlight="-1,1,-1" --glight="1,1,-1" --blight="0,-1,-1"   # flipping z does not mess up just makes a bit dimmer


Reproducing that lighting with world coordinate geometry is being problematic.

* the light position is stored in eye coordinates
* http://stackoverflow.com/questions/15588860/what-exactly-are-eye-space-coordinates





