# === func-gen- : pygame/pygame fgp pygame/pygame.bash fgn pygame fgh pygame
pygame-src(){      echo pygame/pygame.bash ; }
pygame-source(){   echo ${BASH_SOURCE:-$(env-home)/$(pygame-src)} ; }
pygame-vi(){       vi $(pygame-source) ; }
pygame-env(){      elocal- ; }
pygame-usage(){ cat << EOU

PYGAME
=======

Python SDL Wrapper

Source
-------

* https://bitbucket.org/pygame/pygame

Peers
------

* pygame 
* pyglet
* panda3d :doc:/graphics/panda3d/
* pysdl2

pygame benefits from a big community, so its the best one to start with.
Others are more performant and featureful, so make sense later on. 
Thus develop compartmentalized 
keeping G+UI stuff separate from everything else as far as possible.

* http://www.reddit.com/r/Python/comments/15lz1m/pygame_pyglet_something_else_entirely/


SDL
----

* http://en.wikipedia.org/wiki/Simple_DirectMedia_Layer

Simple DirectMedia Layer (SDL) is a cross-platform development library designed
to provide low level access to audio, input devices, and graphics hardware via
OpenGL and Direct3D (i.e. not DirectX). SDL is written in C and is free and
open-source software subject the the requirements of the zlib License since
version 2.0 and GNU Lesser General Public License prior versions.

pygame currently on SDL 1 series
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

From migration to SDL2 guide, https://wiki.libsdl.org/MigrationGuide

mousewheel
^^^^^^^^^^^^

The first change, simply enough, is that the mousewheel is no longer a button.
This was a mistake of history, and we've corrected it in SDL 2.0. Look for
SDL_MOUSEWHEEL events. We support both vertical and horizontal wheels, and some
platforms can treat two-finger scrolling on a trackpad as wheel input, too. You
will no longer receive SDL_BUTTONDOWN events for mouse wheels, and buttons 4
and 5 are real mouse buttons now.

multitouch, gestures
^^^^^^^^^^^^^^^^^^^^^^

Second, there are real touch events now, instead of trying to map this to mouse
input. You can track touches, multiple fingers, and even complex gestures. You
probably want to use those. Refer to SDL_touch.h for a list of these functions,
and look for SDL_Finger* in SDL_events.h.


Trackpad events 
-----------------

* http://stackoverflow.com/questions/10990137/pygame-mouse-clicking-detection

... You can either use the
pygame.mouse.get_pressed method in collaboration with the pygame.mouse.get_pos
(if needed). But please use the mouse click event via a main event loop. The
reason why the event loop is better is due to "short clicks". You may not
notice these on normal machines, but computers that use tap-clicks on trackpads
have excessively small click periods. Using the mouse events will prevent this.


* :google:`pygame trackpad mouse events`





EOU
}
pygame-dir(){ echo $(local-base)/env/pygame/pygame-pygame ; }
pygame-cd(){  cd $(pygame-dir); }
pygame-mate(){ mate $(pygame-dir) ; }
pygame-get(){
   local dir=$(dirname $(pygame-dir)) &&  mkdir -p $dir && cd $dir

}

pygame-docs(){ open /opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/pygame/docs/index.html ; }

