# === func-gen- : graphics/opengl/opengl fgp graphics/opengl/opengl.bash fgn opengl fgh graphics/opengl
opengl-src(){      echo graphics/opengl/opengl.bash ; }
opengl-source(){   echo ${BASH_SOURCE:-$(env-home)/$(opengl-src)} ; }
opengl-vi(){       vi $(opengl-source) ; }
opengl-env(){      elocal- ; }
opengl-usage(){ cat << EOU

OPENGL
=======


* http://www.openglsuperbible.com/2013/12/09/vertex-array-performance/


* http://stackoverflow.com/questions/18814977/using-a-vbo-to-draw-lines-from-a-vector-of-points-in-opengl
* https://www.opengl.org/discussion_boards/showthread.php/176296-glDrawElements-multiple-calls-one-index-array
* http://www.opengl.org/wiki/Vertex_Buffer_Object#Vertex_Buffer_Object


EOU
}
opengl-dir(){ echo $(local-base)/env/graphics/opengl/graphics/opengl-opengl ; }
opengl-cd(){  cd $(opengl-dir); }
opengl-mate(){ mate $(opengl-dir) ; }
opengl-get(){
   local dir=$(dirname $(opengl-dir)) &&  mkdir -p $dir && cd $dir

}
