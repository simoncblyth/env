# === func-gen- : graphics/sdl/sdl fgp graphics/sdl/sdl.bash fgn sdl fgh graphics/sdl
sdl-src(){      echo graphics/sdl/sdl.bash ; }
sdl-source(){   echo ${BASH_SOURCE:-$(env-home)/$(sdl-src)} ; }
sdl-vi(){       vi $(sdl-source) ; }
sdl-env(){      elocal- ; }
sdl-usage(){ cat << EOU

SDL2 : Simple DirectMedia Layer
================================

Simple DirectMedia Layer is a cross-platform development library designed to
provide low level access to audio, keyboard, mouse, joystick, and graphics
hardware via OpenGL and Direct3D. It is used by video playback software,
emulators, and popular games including Valve's award winning catalog and many
Humble Bundle games.  SDL officially supports Windows, Mac OS X, Linux, iOS,
and Android. Support for other platforms may be found in the source code.

* SDL 2.0 and newer are available under the zlib license 
* SDL 1.2 and older are available under the GNU LGPL license 

* http://www.libsdl.org
* http://wiki.libsdl.org/FrontPage

* http://en.wikipedia.org/wiki/Simple_DirectMedia_Layer

* https://wiki.libsdl.org/APIByCategory

  * network absent


Community
-----------

* :google:`sdl tutorial`

* http://lazyfoo.net/tutorials/SDL/index.php
 
  * issues with modern opengl tute

* https://wiki.libsdl.org/Tutorials

Impressions
-------------

* very popular, many resources available

* much more functionality than ultra-lightweight GLFW, maybe bloat 
  but there is some modularity

* http://stackoverflow.com/questions/5736421/sdl-versus-glfw  


DMG binary download
---------------------

* Currently using /Library/Frameworks/SDL2.framework Finder copied from DMG 


macports
--------

Also available from macports::

    delta:~ blyth$ port search libsdl2
    libsdl2 @2.0.3 (devel, multimedia)
        Cross-platform multi-media development API

    libsdl2_image @2.0.0_1 (devel, graphics)
        Add on library for libSDL handling several image formats

    libsdl2_mixer @2.0.0 (audio, devel)
        Audio mixer library for SDL

    libsdl2_net @2.0.0 (devel, net)
        cross-platform networking library

    libsdl2_ttf @2.0.12_1 (devel, graphics)
        add on library for libSDL for rendering TrueType fonts

    Found 5 ports.


lazyfoo tutorial
-----------------

::

    delta:lazyfoo blyth$ clang++ -framework SDL2 01_hello_SDL.cpp -o 01_hello_SDL
    delta:lazyfoo blyth$ ./01_hello_SDL


layzfoo 51_SDL_and_modern_opengl.cpp has bugs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Can compile but draws a blank.

* http://stackoverflow.com/questions/25697203/using-sdl2-glew-trying-to-draw-a-simple-white-square-in-opengl-just-getting


open.gl tutorials
-------------------

* https://open.gl 
* https://open.gl/context compares use of SFML, SDL, GLFW





EOU
}
sdl-dir(){ echo $(local-base)/env/graphics/sdl ; }
sdl-sdir(){ echo $(env-home)/graphics/sdl ; }
sdl-cd(){  cd $(sdl-dir); }
sdl-scd(){ cd $(sdl-sdir); }

sdl-mate(){ mate $(sdl-dir) ; }
sdl-get(){
   local dir=$(dirname $(sdl-dir)) &&  mkdir -p $dir && cd $dir

}

sdl-dmg-url(){ echo http://www.libsdl.org/release/SDL2-2.0.3.dmg ; }
sdl-dmg-get(){
   local dir=$(sdl-dir) &&  mkdir -p $dir && cd $dir
   local url=$(sdl-dmg-url)
   local dmg=$(basename $url)
   [ ! -f $dmg ] && curl -L -O $url

   cat << EOM

* Opening dmg $dmg with finder...

* Drag the SDL2.framework from Finder window to /Library/Frameworks/

* Readme recommends

  * http://www.openscenegraph.org/projects/osg/wiki/Support/Tutorials/MacOSXTips


EOM

   open $dmg

}

sdl-framework-sign(){
   cd /Library/Frameworks/SDL2.framework/
   codesign -f -s - SDL2
}


sdl-framework-include(){
   perl -pi -e 's,include <SDL.h>,include <SDL2/SDL.h>,' ${1:-code.cpp} 
}

sdl-example-dir(){   echo $(sdl-sdir)/lazyfoo ; }
sdl-example-cd(){    cd  $(sdl-example-dir) ; }
sdl-example-make(){  

   local cpp=${1:-example.cpp}
   local bin=${cpp/.cpp}
   [ ! -f "$cpp" ] && echo expecting a file eg  $cpp && return  
   local cmd="clang++ -framework SDL2 $cpp -o $bin"
   echo $msg $cmd
   eval $cmd
   
}



sdl-example-glew-make(){
   local cpp=${1:-51_SDL_and_modern_opengl.cpp}
   local bin=${cpp/.cpp}
   [ ! -f "$cpp" ] && echo expecting a file eg  $cpp && return  

   glew-
   local glewbase=$(glew-idir)
   local cmd="clang++ -framework SDL2 -I$glewbase/include -L$glewbase/lib -lGLEW -framework OpenGL $cpp -o $bin"
   echo $cmd
   eval $cmd
}




