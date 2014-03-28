daeviewgl debugging
====================

Often loose geometry, ie after making some changes 
find that only get blank screens.  

Tricks to work out whats gone wrong
------------------------------------

#. switch to line mode with `--nolight --line`
#. vary near, far, yfov
#. check scaled mode

    daeviewgl.py -t +0 --eye "2,2,2"


Issue
------

Targetting outer volumes draw blanks
--------------------------------------

::

    daeviewgl.py -n 3153:12230 -t "" --nolight --line     #  blank
    daeviewgl.py -n 3153:12230 -t +0 --nolight --line     # 
    daeviewgl.py -n 3153:12230 -t +6 --nolight --line     #  +0 to +6 all draw blanks
    daeviewgl.py -n 3153:12230 -t 3159 --nolight --line   # absolute equivalent matches

    daeviewgl.py -n 3153:12230 -t +7 --nolight --line     #  first visible volume 
    daeviewgl.py -n 3153:12230 -t 3160 --nolight --line      

    daeviewgl.py -n 3153:12230 -t -10                     # last visible
    daeviewgl.py -n 3153:12230 -t -9                      # same issue at end of volume list 
    daeviewgl.py -n 3153:12230 -t -1       

    daeviewgl.py -n 3153:12230 -t -9 --wlight 0           # lights at infinity dont help 


#. maybe lights contained inside some geometry, so cannot see anything when go outside that volume ? 

   * but light is off ?

#. wierd, via remote control can succeed to see the outer volumes when 

   * startup in situation where see something
   * seems startup is setting the light positions ? which infulence even `--nolight` ?
   * the issue is of handling scale changes, need better feedback into title bar 


Near
-----

Need crazy small near in many situations, why? a scaling problem still::

    daeviewgl.py -n 3153:12230 -t "+0" --nearclip 1e-6,1e6
    daeviewgl.py -n 3153:12230 -t "" --nearclip 1e-6,1e6


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

    daeviewgl.py


target mode
~~~~~~~~~~~~

::

    daeviewgl.py -t "" 
    daeviewgl.py -t "" --nolight --line

    daeviewgl.py -t +0
    daeviewgl.py -t +0    --nolight --line

    daeviewgl.py -t +1000 --nolight --line   
 


Scenarios
----------

detail view
~~~~~~~~~~~~

View targetting a single small piece of geometry (eg a PMT) within context 
of many other such pieces of geometry all within containing geometry (eg the AD) 

::

    daeviewgl.py -n 3153:6000 -t 5000
    daeviewgl.py -n 3153:6000 -t 5000 --nolight --line      # linemode

outside view
~~~~~~~~~~~~~~

::

    daeviewgl.py -n 3153:6000 -t ""
    daeviewgl.py -n 3153:6000 -t 3153 --nolight --line  





