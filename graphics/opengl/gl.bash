# === func-gen- : graphics/opengl/gl fgp graphics/opengl/gl.bash fgn gl fgh graphics/opengl
gl-src(){      echo graphics/opengl/gl.bash ; }
gl-source(){   echo ${BASH_SOURCE:-$(env-home)/$(gl-src)} ; }
gl-vi(){       vi $(gl-source) ; }
gl-env(){      elocal- ; }
gl-usage(){ cat << EOU

OpenGL
=======

Modern Open GL 3.x and 4 resources
-------------------------------------

* http://www.swiftless.com/opengl4tuts.html

* https://www.opengl.org/wiki/Related_toolkits_and_APIs

  * GLFW http://www.glfw.org
  * SDL
  * SFML
  * Allegro http://alleg.sourceforge.net/






EOU
}
gl-dir(){ echo $(local-base)/env/graphics/opengl/graphics/opengl-gl ; }
gl-cd(){  cd $(gl-dir); }
gl-mate(){ mate $(gl-dir) ; }
gl-get(){
   local dir=$(dirname $(gl-dir)) &&  mkdir -p $dir && cd $dir

}
