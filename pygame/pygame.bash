# === func-gen- : pygame/pygame fgp pygame/pygame.bash fgn pygame fgh pygame
pygame-src(){      echo pygame/pygame.bash ; }
pygame-source(){   echo ${BASH_SOURCE:-$(env-home)/$(pygame-src)} ; }
pygame-vi(){       vi $(pygame-source) ; }
pygame-env(){      elocal- ; }
pygame-usage(){ cat << EOU

PYGAME
=======

Python SDL Wrapper

SDL
----

* http://en.wikipedia.org/wiki/Simple_DirectMedia_Layer

Simple DirectMedia Layer (SDL) is a cross-platform development library designed
to provide low level access to audio, input devices, and graphics hardware via
OpenGL and Direct3D (i.e. not DirectX). SDL is written in C and is free and
open-source software subject the the requirements of the zlib License since
version 2.0 and GNU Lesser General Public License prior versions.


EOU
}
pygame-dir(){ echo $(local-base)/env/pygame/pygame-pygame ; }
pygame-cd(){  cd $(pygame-dir); }
pygame-mate(){ mate $(pygame-dir) ; }
pygame-get(){
   local dir=$(dirname $(pygame-dir)) &&  mkdir -p $dir && cd $dir

}

pygame-docs(){ open /opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/pygame/docs/index.html ; }

