g4daeview.py
=============

.. contents:: :local:

Usage Tips
------------

#. Exit other apps that make heavy use of GPU acceleration whilst running
   g4daeview.py (and especially when launching Chroma raycasts)

   * Google Chrome snags GPU resources far more that Safari, 
   * Uncheck `Use hardware acceleration when available` in `Chrome > Settings > Advanced Settings [System]`
   * and restart Chrome for it to take effect

Usage text
-------------

On pressing "U" usage text describing the actions of each key 
are written to stdout.  (TODO: display to screen)


Receive ChromaPhotonList from Remote Geant4 
--------------------------------------------- 

ChromaPhotonList instances are transported using ZMQRoot 

* http://dayabay.ihep.ac.cn/tracs/dybsvn/browser/dybgaudi/trunk/Utilities/ZMQRoot

`ZMQRoot` package uses ROOT TMessage (de)serialization 
and ZeroMQ messaging to implement a simple API::

   class ZMQRoot {
   public:
         ZMQRoot(const char* envvar);
         void SendObject(TObject* obj);
         TObject* ReceiveObject();

A three node network topology is currently used:

*client*
     `nuwa.py/Geant4` process
*worker*
     `g4daeview.py/Chroma` process 
*broker*
     `zmq_broker.sh` process

Usage steps:

* check if the ZMQ broker is running on belle7, if not launch it::

    delta:~ blyth$ ssh N
    [blyth@belle7 ~]$ pgrep zmq_broker
    [blyth@belle7 ~]$ zmq_broker.sh

* launch the worker, envvar defaults configure connection to broker on belle7::

    delta:~ blyth$ g4daeview.sh

* launch the client on belle7::

    [blyth@belle7 ~]$ which csa.sh   # CSA for ChromaStackAction
    ~/env/bin/csa.sh
    [blyth@belle7 ~]$ csa.sh 


The `csa.sh` job within its ChromaStackAction collects photons 
into a `ChromaPhotonList` and ready for transport with `ZMQRoot`.


Load/Save ChromaPhotonList 
-------------------------------------

Once a CPL has arrived via ZMQ, it can be saved to file with::

    delta:~ blyth$ udp.py --save /usr/local/env/tmp/1.root

Subsequently can load the root file at launch with::

    delta:g4daeview blyth$ g4daeview.sh --load /usr/local/env/tmp/1.root

Or loading can be done interactively with `udp.py` to change or set the loaded event::

    delta:~ blyth$ udp.py --load /usr/local/env/tmp/1.root 

The objects loaded or saved from root files can also have an associated *key* using 
option `--key WHATEVER` the default key is **CPL**.

Launch arguments only support a single `--load`, but interactive arguments 
support multiple loads allowing multiple objects to be loaded.

* TODO: dont replace but concatenate drawing of multiple loaded objects   

Screen Captures and apache publishing
---------------------------------------

Save any screencaptures directly into apache htdocs using::

    g4(){
        apache-
        g4daeview.sh --outdir=$(apache-htdocs)/env/geant4/geometry/collada/g4daeview --load 1 $*
    }


Captures can be remotely invoked using::

   udp.py --screencapture


Grab Screen Captures including GLUT menus
--------------------------------------------

The above screencapture techniqies exclude any GLUT menus that are appearing. 
To include those use:

* `Grab.app > Capture > Timed Screen`  screen is captured 10s after starting timer, allowing to prepare menus
* the image is opened in Grab.app and can be saved in a `.tiff`



Subsequently propagate to remote apache with::

   env-htdocs-rsync C2

TODO: adopt config file for settings like outdir (maybe `~/.g4daeview/config.cfg`) 


Partial geometry
------------------

Partial geometry can be specified listwise or treewise, for example::

    # listwise
    g4daeview.py -p dyb -g 3153:      # skips Universe, rock and RPC, but includes everything in the pool  
    g4daeview.py -p dyb -g 6473:      # skips the ADs 
    g4daeview.py -p dyb -g 4:3146     # RPC

    # treewise shortform
    g4daeview.py -p dyb -g 3147+      # 
    g4daeview.py -p dyb -g 3152+      # without radslabs
    g4daeview.py -p dyb -g 3154-      # includes only SST and nodes beneath that in the tree, ie the contents of SST
    g4daeview.py -p dyb -g 4536+ -n2  # children of a volume (calibration dome) excluding the volume itself, near is specified to avoid near clipping

    # treewise longform
    g4daeview.py -p dyb -g 3154_1.5   # specify min/max recursion depth from the basenode
    g4daeview.py -p dyb -g 2_0.0        # just volume 2
    g4daeview.py -p dyb -g 2_1.1        # just immediate children of volume 2 

    # combination form 
    g4daeview.py -p dyb -g 3153+,4813+
    g4daeview.py -p dyb -g 3153_1.2,4813_1.2
    g4daeview.py -p dyb -g 3153_0.2,4813_0.2,6473: --with-chroma    # smoke and mirrors, looks like default "3153:" but with much fewer volumes

    g4daeview.py -p dyb -g 3153_0.2,4813_0.2,6473: --with-chroma --size 640,480 --launch 1,1,1


TIP: to determine node tree indices click on the solids in the viewer and
use daeserver urls like the below to list the tree::

    http://belle7.nuu.edu.tw/dae/tree/4813___2.txt?ancestors=1

The partial geometry specified is also used for Chroma raycasting, which 
is a simple way to make raycast rendering faster.

Target Mode
-------------

Target mode presents many volumes but targets one by 
orienting the initial viewpoint with respect to the target using 
units based on the extent of the target and axis directions

Identify target via relative to node list (starting with `+` or `-`) or absolute addressing::

    g4daeview.py -g 3153:12230 -t -300 
    g4daeview.py -g 3153:12230 -t +10
       
       # target relative to the node list 

    g4daeview.py -g 3153:12230 -t +0       # relative 
    g4daeview.py -g 3153:12230 -t 3153     # absolute equivalent 

    g4daeview.py -t +0      

       # when using a sensible default node list, this is convenient 

    g4daeview.py -g 3153:12230 -t 5000 --eye="-2,-2,-2"

       # control the starting eyepoint relative to the target 


near/far clipping controls
-----------------------------

When you approach too closely to some geometry it will disappear due to 
near clipping. Similarly distant geometry can disappear from far clipping.
To control the near/far clipping planes:

* press "N" while dragging up/down to change "near" distance
* press "F" while dragging up/down to change "far" distance

To illustrate the viewing frustum (square pyramid chopped at near/far planes
with "eye" at the apex) and near/far planes press "K" to switch on 
markers and trackball away from the view into order to look back 
at its frustum. Also change "near" and "far" to see how that 
changes the depth planes.

ISSUES
~~~~~~~~~
 
Somehow changing "near" sometimes acts to change "far" clipping. 
Possibly this is due to limited depth buffering, the issue 
seems less prevalent with less extreme "near" and "far" values.

solid picking
---------------

#. Clicking pixels with mouse/trackpad, yields an (x,y,z) screen position 
   the z value comes from the depth buffer representing the  position of nearest surface.
#. An unprojection transforms the screen space coordinate into world space.
#. This coordinate is then used to determine the list of solids which 
   contain the point within their bounding box. The solid indices, names and 
   extents in mm are written to the screen.
#. The smallest solid is regarded as "picked". Key "O" toggles high-lighting 
   of picked solids with wireframe spheres.

TODO
~~~~~

Make more use of this, eg to display material/surface properties, 
position in heirarchy 


placemarks
-------------

The commandline to return to a viewpoint and camera configuration
is written to stdout on exiting or on pressing "W".

markers
----------

Switch on markers with "K", the look point is illustrated with a 
wireframe cube with wireframe sphere inside. Also the frustum of the current view 
excluding any offset trackball rotation + translation and raycast direction/origin
are illustrated.

 
Viewpoint Bookmarks
---------------------

Number keys are used to create and visit bookmarks. 
While pressing a number key 1-9 click on a solid to create a bookmark, 
the view adopts the coordinate frame corresponding to the solid clicked. 
Subsequently pressing number keys 0-9 visit the bookmarks created, 
and pressing SPACE updates the current bookmark (last one created/visited) 
to bake any offsets made from the view into the view. 
Bookmark 0 is created at startup for the initial viewpoint.

A bookmark comprises: 

* a solid (or the entiremesh), which defines the view coordinate system. 
  Unit of length is the extent of the solid 
* "eye" position, eg -2,-2,0  
* "look" position, eg 0,0,0 : about which trackball rotations are applied
* "up" direction, eg 0,0,1 

Note that trackball translations/rotations do not update the "view", 
although they do of course update what you see. To solidify trackballing
offsets into the current view press SPACE. 

* drag around to rotate about the "look" point using a virtual trackball,
  XY positions are projected onto virtual sphere trackball, which allow
  offset rotations to be obtained via some Quaternion math   
* press "X" while dragging around to translate in screen XY direction 
* press "Z" while dragging up/down to tranlate in screen Z direction (in/out)

All bookmarks other than bookmark zero which corresponds to the launch viewpoint 
are persisted at exit into a "bookmarks_%(path)s.cfg" file in the working directory, 
where the path is filled in with in launch path argument. This allows separate 
bookmarks to be maintained per site. 
A subsequent session from the same directory re-loads the bookmarks.

ISSUES
~~~~~~~

#. when viewing partial geometry bookmarks which refer to volumes that are not present 
   are not loaded, so bookmarks set when using more complete geometry will be lost.
   Workaround is to launch from a different directories for different
   geometries or use `--bookmarks path` option.
   Solution of incorporating geometry spec into the bookmarks name, seems clumsy.

#. perhaps persist the exiting viewpoint into bookmark-0 ? 



Animation
----------

interpolate between bookmarks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Press "B" to setup an animation that linearly interpolates between the 
bookmarked views starting at the current bookmark. Two or more bookmarks
are required.  To change the animation first update the bookmarks 
and then press "B" again.

orbiting mode
~~~~~~~~~~~~~~

Press "V" to setup a flyaround or orbit mode for the current bookmark.
The initial "look" direction is tangential, so you might need to turn inwards 
using the trackball controls to see the geometry. 

ISSUES: rotation point not where intended, makes difficult to use

animation control
~~~~~~~~~~~~~~~~~

Following setup of bookmark or orbit animations, pressing "M" will toggle 
the animation. Speed of animation can be adjusted using the right/left arrow keys.
During animation trackball translation/rotation can still be used to adjust the effective viewpoint. 
Also most other controls can still be used during the animation, such as near/far clipping or 
switching to Chroma raycast rendering.

TODO: key to reverse animation 


remote control
---------------

A subset of the commandline launch options can be sent over the network to the running 
application. This allows numerical control of viewpoint and camera parameters.::

   udp.py -t 7000 --eye=10,0,0 --look=0,0,0 --up=0,0,1
   udp.py -t 7000_10,0,0_0,0,0_0,0,1                    # equivalent short form

The viewpoint is defined by the `eye` and `look` positions and the `up` direction, which 
are provided in the coordinate frame of the target solid. NB rotations are performed about the 
look position, that is often set to 0,0,0 corresponding to the center of the solid. 
The "K" key toggle markers indicating the eye and look position. 

The options that are accepted interactively are marked with "I" in the options list::

    daeview.sh --help


raycast launch control
------------------------

Press "R" to toggle raycast mode.

To avoid GPU panics/system crashes

* subsequent raycast launches are aborted when launch times exceed *max-time* cutoff 
* launch configuration is controlled by eg *launch=3,2,1* and *block=8,8,1* 
  options which configure 2D launch 
 
* raycast launches tyically use 2D pixel thread blocks, 
  some speedups achieved by moving from line of pixels to 2D regions
  in order for the work within a warp of 32 threads to be more uniform 


interactive switch to metric pixels presentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The chroma raycast metrics available for display must be defined at 
launch with eg::

   g4daeview.py --metric time/tri/node  

The restricted flexibility is due to needing to compile
the kernel to change the metric. This is to avoid little 
used branching in the kernel.

Kernel flags can be controlled by remote control, eg::

   udp.py --flags 15,0    # does 15 controls bit shift, here "metric >> 15"  


screen capture
----------------

Pressing "E" will create a screen capture and write a timestamp dated .png 
to the current directory.

movie capture
--------------

Not implemented, as find that on OSX can simply use `QuickTime Player.app` 

* `File > New Screen Recording` to create a very large .mov (~1GB for ~2min) 
* `File > Export ...` to compress .mov to .m4v 


Parallel/Orthographic projection
----------------------------------

Press "P" to toggle between orthographic/parallel projection and the default
perspective projection. 

* "Z" to translate eye point in/out

  As parallel projection corresponds to the view from infinity it is
  somewhat paradoxical that translating in Z has any effect, however
  it does so indirectly via changing where the near clipping plane falls 

* "N" to change near
* "Y" to change yfov 
* "F" to change far 

To "enter" geometry while in parallel, use small FOV (eg 5 degrees) 
and vary near and Z in order to clip the volumes.


TODO: Get Chroma raycast to work in parallel projection mode, need to 
      come up with the matrix and probably change the kernel.
 

Presentation
--------------

::

    g4daeview.py -g 4998:6000

      # default includes lights, fill with transparency 

    g4daeview.py -g 4998:6000 --line

      # adding wireframe lines slows rendering significantly,toggle lines with "L"

    g4daeview.py -g 4998 --nofill

       # without polygon fill the lighting/transparency has no effect, toggle face fill with "A"

    g4daeview.py -g 4998 --nofill 

       # blank white 

    g4daeview.py -g 4900:5000,4815 --notransparent

       # see the base of the PMTs poking out of the cylinder when transparency off

    g4daeview.py -g 4900:5000,4815 --rgba .7,.7,.7,0.5

       # changing colors, especially alpha has a drastic effect on output

    g4daeview.py -g 3153:6000

       # inside the pool, 2 ADs : navigation is a challenge, its dark inside

    g4daeview.py -g 6070:6450

       # AD structure, shows partial radial shield

    g4daeview.py -g 6480:12230 

       # pool PMTs, AD support, scaffold?    when including lots of volumes switching off lines is a speedup

    g4daeview.py -g 12221:12230 

       # rad slabs

    g4daeview.py -g 2:12230 

       # full geometry, excluding only boring (and large) universe and rock 

    g4daeview.py -g 3153:12230

       # skipping universe, rock and RPC makes for easier inspection inside the pool







