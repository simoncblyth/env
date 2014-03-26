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


