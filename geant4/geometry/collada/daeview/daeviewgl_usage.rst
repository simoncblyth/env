daeviewgl.py
=============

Parallel/Orthographic projection
----------------------------------

In parallel projection, there is effectively no z direction (its
as if viewing from infinity) so varying z has no effect.  Instead
to control view change near and yfov.  It can be difficult 
to "enter" geometry while in parallel, to do so try very small yfov (5 degrees) 
and vary near.

Observations
--------------

#. toggling between parallel/perspective can be dis-orienting, change near/yfov to get the desired view  


Up/down views are problematic
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When `look-eye` and `up` are in same direction see nothing::

    daeviewgl.py -n 5000:6000 -t "" --eye="0,0,-2" --look="0,0,0" --up="0,0,1"        # nowt
    daeviewgl.py -n 5000:6000 -t "" --eye="0,0,2" --look="0,0,0" --up="0,0,1" 

Tweaking eye or up regain the geometry::

    daeviewgl.py -n 5000:6000 -t "" --eye="0,0,-2" --look="0,0,0" --up="0,0.001,1"   
    daeviewgl.py -n 5000:6000 -t "" --eye="0,0.001,-2" --look="0,0,0" --up="0,0,1"    

    # nodes 5000:6000 corresponds to several rings of PMTs, making for a good simple test geometry 
    # the view is of circles of PMTs from above/below


* TODO: automate the tweaking



More Bizarrness
---------------

Off into outerspace and back::

    (chroma_env)delta:daeview blyth$ grep ESR solids.txt
    1274  DAESolid v 578  t 1188  n 578   : 4427 __dd__Geometry__AdDetails__lvTopRefGap--pvTopESR0xab4bd50.0   
    1277  DAESolid v 330  t 688  n 330   : 4430 __dd__Geometry__AdDetails__lvBotRefGap--pvBotESR0xae4eda0.0   
    2934  DAESolid v 578  t 1188  n 578   : 6087 __dd__Geometry__AdDetails__lvTopRefGap--pvTopESR0xab4bd50.1   
    2937  DAESolid v 330  t 688  n 330   : 6090 __dd__Geometry__AdDetails__lvBotRefGap--pvBotESR0xae4eda0.1   
    (chroma_env)delta:daeview blyth$ 
    (chroma_env)delta:daeview blyth$ ./udp.py "-j 4427_-5,2,2:"
    sending -j 4427_-5,2,2: 

Its because its a 4.5 disk shape, look from above::

    udp.py "-t 4427_0,0.001,-2"
    udp.py "-t 4427_0,0.001,5"

    ./udp.py "-t 4427_0,0.001,-5"   # view upwards from bottom of AD 



Usage Examples
---------------

Target Mode
~~~~~~~~~~~~~

Identify target via relative to node list (starting with `+` or `-`) or absolute addressing::

    daeviewgl.py -n 3153:12230 -t -300 
    daeviewgl.py -n 3153:12230 -t +10
       
       # target relative to the node list 

    daeviewgl.py -n 3153:12230 -t +0       # relative 
    daeviewgl.py -n 3153:12230 -t 3153     # absolute equivalent 

    daeviewgl.py -t +0      

      # when using a sensible default node list, this is convenient 



Animation
~~~~~~~~~~~

::

    (chroma_env)delta:daeview blyth$ ./daegeometry.py > solids.txt
    INFO:env.geant4.geometry.collada.daenode:DAENode.parse pycollada parse /Users/blyth/env/geant4/geometry/materials/g4_00.dae 
    INFO:__main__:Flattening 9077 DAESolid into one DAEMesh...
    INFO:__main__:DAEMesh v 1234834  t 2438640  n 1234834 
    (chroma_env)delta:daeview blyth$ vi solids.txt 
    (chroma_env)delta:daeview blyth$ grep ball solids.txt
    (chroma_env)delta:daeview blyth$ grep Ball solids.txt
    1369  DAESolid v 267  t 528  n 267   : 4522 __dd__Geometry__CalibrationSources__lvWallLedSourceAssy--pvWallLedDiffuserBall0xab71f78.0   
    1372  DAESolid v 267  t 528  n 267   : 4525 __dd__Geometry__CalibrationSources__lvWallLedSourceAssy--pvWallLedDiffuserBall0xab71f78.1   
    1375  DAESolid v 267  t 528  n 267   : 4528 __dd__Geometry__CalibrationSources__lvWallLedSourceAssy--pvWallLedDiffuserBall0xab71f78.2   
    1389  DAESolid v 267  t 528  n 267   : 4542 __dd__Geometry__CalibrationSources__lvLedSourceShell--pvDiffuserBall0xabe00c8.0   
    1477  DAESolid v 267  t 528  n 267   : 4630 __dd__Geometry__CalibrationSources__lvLedSourceShell--pvDiffuserBall0xabe00c8.1   
    1559  DAESolid v 267  t 528  n 267   : 4712 __dd__Geometry__CalibrationSources__lvLedSourceShell--pvDiffuserBall0xabe00c8.2   
    3029  DAESolid v 267  t 528  n 267   : 6182 __dd__Geometry__CalibrationSources__lvWallLedSourceAssy--pvWallLedDiffuserBall0xab71f78.3   
    3032  DAESolid v 267  t 528  n 267   : 6185 __dd__Geometry__CalibrationSources__lvWallLedSourceAssy--pvWallLedDiffuserBall0xab71f78.4   
    3035  DAESolid v 267  t 528  n 267   : 6188 __dd__Geometry__CalibrationSources__lvWallLedSourceAssy--pvWallLedDiffuserBall0xab71f78.5   
    3049  DAESolid v 267  t 528  n 267   : 6202 __dd__Geometry__CalibrationSources__lvLedSourceShell--pvDiffuserBall0xabe00c8.3   
    3137  DAESolid v 267  t 528  n 267   : 6290 __dd__Geometry__CalibrationSources__lvLedSourceShell--pvDiffuserBall0xabe00c8.4   
    3219  DAESolid v 267  t 528  n 267   : 6372 __dd__Geometry__CalibrationSources__lvLedSourceShell--pvDiffuserBall0xabe00c8.5   
    (chroma_env)delta:daeview blyth$ 
    (chroma_env)delta:daeview blyth$ daeviewgl.py -t +0 -j +1369,+1372,+1375,+1389,+1477,+1559,+3029,+3032,+3035,+3049,+3137,+3219 --near 1e-5


Reflectors
------------

::

    (chroma_env)delta:daeview blyth$ grep TopReflector solids.txt
    1272  DAESolid v 296  t 608  n 296   : 4425 __dd__Geometry__AD__lvOIL--pvTopReflector0xab22490.0   
    1273  DAESolid v 296  t 608  n 296   : 4426 __dd__Geometry__AdDetails__lvTopReflector--pvTopRefGap0xabcc228.0   
    2932  DAESolid v 296  t 608  n 296   : 6085 __dd__Geometry__AD__lvOIL--pvTopReflector0xab22490.1   
    2933  DAESolid v 296  t 608  n 296   : 6086 __dd__Geometry__AdDetails__lvTopReflector--pvTopRefGap0xabcc228.1   
    (chroma_env)delta:daeview blyth$ 

    ./udp.py "-t 4425_0,0.001,-2_-2,-2,0
"   # looking up at top reflector


Bizarre
~~~~~~~~

Issues when small extent ?

::

    daeviewgl.py -t +1369

    daeviewgl.py -t +1369 --eye=0,0.001,20    # small ball and cylinder



Interpolation
~~~~~~~~~~~~~~~~


Expected yoyo, just get fall::

    daeviewgl.py -t 8153 --eye="2,2,40" --look="2,2.001,0" -j +0_2,2,-40:+0_2,2,40    



Very long shapes are problematic::

    daeviewgl.py -t 4522 -j 4522_0,5,0:4522_5,0,0:4522_0,0.001,5 --near 1e-6 --far 1e6

    daeviewgl.py -n 4522,4525,4528,4542,4630,4712

    daeviewgl.py -n 4522,4525,4528,4542,4630,4712 -t "" -j 4522:4525:4528:4542:4630:4712

    daeviewgl.py -n 4522:4712 -t 4522

    daeviewgl.py -t 4522 -j 4522_0,5,0:4522_5,0,0:4522_0,0.001,5 --near 0.01

    daeviewgl.py -t 4522 -j 4522_0,5,0:4522_5,0,0:4522_0,0.001,5 



Interpolation Jumps
~~~~~~~~~~~~~~~~~~~


::

    daeviewgl.py -t +1000 -j +1000_2,2,2:+1000_2,2,10

    daeviewgl.py -t 4522 -j 4522_0,5,0:4522_5,0,0:4522_0,0.001,5



Presentation
~~~~~~~~~~~~~


::

    daeviewgl.py -n 4998:6000

      # default includes lights, fill with transparency 

    daeviewgl.py -n 4998:6000 --line

      # adding wireframe lines slows rendering significantly

    daeviewgl.py -n 4998 --nofill

       # without polygon fill the lighting/transparency has no effect

    daeviewgl.py -n 4998 --nofill 

       # blank white 

    daeviewgl.py -n 4900:5000,4815 --notransparent

       # see the base of the PMTs poking out of the cylinder when transparency off

    daeviewgl.py -n 4900:5000,4815 --rgba .7,.7,.7,0.5

       # changing colors, especially alpha has a drastic effect on output

    daeviewgl.py -n 4900:5000,4815 --ball 90,0,2,3

       # primitive initial position control using trackball arguments, theta,phi,zoom,distance

    daeviewgl.py -n 3153:6000

       # inside the pool, 2 ADs : navigation is a challenge, its dark inside

    daeviewgl.py -n 6070:6450

       # AD structure, shows partial radial shield

    daeviewgl.py -n 6480:12230 

       # pool PMTs, AD support, scaffold?    when including lots of volumes switching off lines is a speedup

    daeviewgl.py -n 12221:12230 

       # rad slabs

    daeviewgl.py -n 2:12230 

       # full geometry, excluding only boring (and large) universe and rock 

    daeviewgl.py -n 3153:12230

       # skipping universe, rock and RPC makes for easier inspection inside the pool

    daeviewgl.py  -n 3153:12230 -t 5000 --eye="-2,-2,-2"

       # target mode, presenting many volumes but targeting one and orienting viewpoint with 
       # respect to the target using units based on the extent of the target and axis directions
       # from the world frame
       #
       # long form --eye="..." is needed as the value starts with "-"


