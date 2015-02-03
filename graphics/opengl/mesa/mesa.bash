# === func-gen- : graphics/opengl/mesa/mesa fgp graphics/opengl/mesa/mesa.bash fgn mesa fgh graphics/opengl/mesa
mesa-src(){      echo graphics/opengl/mesa/mesa.bash ; }
mesa-source(){   echo ${BASH_SOURCE:-$(env-home)/$(mesa-src)} ; }
mesa-vi(){       vi $(mesa-source) ; }
mesa-env(){      elocal- ; }
mesa-usage(){ cat << EOU


Mesa 3D : Open Source Implementation of OpenGL specification
=================================================================
 
* http://www.mesa3d.org



EOU
}
mesa-dir(){ echo $(local-base)/env/graphics/opengl/mesa/graphics/opengl/mesa-mesa ; }
mesa-cd(){  cd $(mesa-dir); }
mesa-mate(){ mate $(mesa-dir) ; }
mesa-get(){
   local dir=$(dirname $(mesa-dir)) &&  mkdir -p $dir && cd $dir

}
