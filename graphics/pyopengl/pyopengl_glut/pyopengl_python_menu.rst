PyOpenGL Python Menu
=====================

Thoughts
---------

#. coming out of GLUT.framework ?
#. developments in this realm inevitably system specific, BUT nicer menus 
   that the pure GLUT ones 

Observations
-------------

When using *g4daeview.py* (a pyopengl app built on glumpy) an "exec" process
is shown on OSX GUI process list (from `cmd-tab`). Selecting that 
reveals a system menu heirarchy entitled "python". 

The `File > Preferences...` is titled `GLUT Preferences` and includes tabs:

* Initialization
* Mouse
* Joystick
* Spaceball

Initialization
----------------

* Default Geometry: 300 x 300 @ -1 x -1
  (this explains the small window that appears very briefly at startup)  


Mouse
-------

::

   3 button mouse detected.

   [checkbox] Enable Button Emulation 

      Right Button Modifier:   *Control*/Option/Command/Shift
      Middle Button Modifier:  Control/*Option*/Command/Shift

 


