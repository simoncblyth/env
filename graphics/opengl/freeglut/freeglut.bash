# === func-gen- : graphics/opengl/freeglut/freeglut fgp graphics/opengl/freeglut/freeglut.bash fgn freeglut fgh graphics/opengl/freeglut
freeglut-src(){      echo graphics/opengl/freeglut/freeglut.bash ; }
freeglut-source(){   echo ${BASH_SOURCE:-$(env-home)/$(freeglut-src)} ; }
freeglut-vi(){       vi $(freeglut-source) ; }
freeglut-env(){      elocal- ; }
freeglut-usage(){ cat << EOU





EOU
}
freeglut-dir(){ echo $(local-base)/env/graphics/opengl/freeglut/FreeGLUT ; }
freeglut-cd(){  cd $(freeglut-dir); }
freeglut-mate(){ mate $(freeglut-dir) ; }
freeglut-get(){
   local dir=$(dirname $(freeglut-dir)) &&  mkdir -p $dir && cd $dir

   git clone https://github.com/dcnieho/FreeGLUT.git

}
