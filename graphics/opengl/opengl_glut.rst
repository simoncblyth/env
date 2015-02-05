OpenGL GLUT 
=============

Sources
--------

* http://freeglut.sourceforge.net/docs/api.php

* http://sourceforge.net/projects/freeglut/  yuck SF
* https://github.com/dcnieho/FreeGLUT



OSX supporting two finger trackpad scroll
------------------------------------------

* :google:`opengl glut osx`

* http://iihm.imag.fr/blanch/software/glut-macosx/

* http://freeglut.sourceforge.net/docs/api.php


Chroma camera gets pygame MOUSEBUTTONDOWN events, can i get the same from GLUT ?
-----------------------------------------------------------------------------------

SDL 1.0 to 2.0 migration
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://wiki.libsdl.org/MigrationGuide#Input


The first change, simply enough, is that the mousewheel is no longer a button.
This was a mistake of history, and we've corrected it in SDL 2.0. Look for
SDL_MOUSEWHEEL events. We support both vertical and horizontal wheels, and some
platforms can treat two-finger scrolling on a trackpad as wheel input, too. You
will no longer receive SDL_BUTTONDOWN events for mouse wheels, and buttons 4
and 5 are real mouse buttons now.


* http://www.programming-techniques.com/2012/01/glut-tutorial-how-to-detect-mouse-click.html
* http://www.lighthouse3d.com/opengl/glut/index.php3?9


* http://stackoverflow.com/questions/14378/using-the-mouse-scrollwheel-in-glut


From comment in glumpy source

* https://github.com/nanoant/osxglut





