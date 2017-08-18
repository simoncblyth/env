# === func-gen- : graphics/sfml/sfml fgp graphics/sfml/sfml.bash fgn sfml fgh graphics/sfml
sfml-src(){      echo graphics/sfml/sfml.bash ; }
sfml-source(){   echo ${BASH_SOURCE:-$(env-home)/$(sfml-src)} ; }
sfml-vi(){       vi $(sfml-source) ; }
sfml-env(){      elocal- ; }
sfml-usage(){ cat << EOU


SFML : Simple and Fast Multimedia Library
==========================================

* http://www.sfml-dev.org
* http://www.sfml-dev.org/learn.php

* https://www.sfml-dev.org/tutorials/2.4/window-opengl.php


SFML provides a simple interface to the various components of your PC, to ease
the development of games and multimedia applications. It is composed of five
modules: system, window, graphics, audio and network.




EOU
}
sfml-dir(){ echo $(local-base)/env/graphics/sfml/graphics/sfml-sfml ; }
sfml-cd(){  cd $(sfml-dir); }
sfml-mate(){ mate $(sfml-dir) ; }
sfml-get(){
   local dir=$(dirname $(sfml-dir)) &&  mkdir -p $dir && cd $dir

}
