# === func-gen- : graphics/opengl/nature/nature fgp graphics/opengl/nature/nature.bash fgn nature fgh graphics/opengl/nature
nature-src(){      echo graphics/opengl/nature/nature.bash ; }
nature-source(){   echo ${BASH_SOURCE:-$(env-home)/$(nature-src)} ; }
nature-vi(){       vi $(nature-source) ; }
nature-env(){      elocal- ; }
nature-usage(){ cat << EOU

OpenGL Instance Culling Using Geometry Shaders Demo
=====================================================

* http://rastergrid.com/blog/2010/02/instance-culling-using-geometry-shaders/


EOU
}
nature-dir(){ echo $(local-base)/env/graphics/opengl/nature ; }
nature-cd(){  cd $(nature-dir); }
nature-c(){   cd $(nature-dir); }
nature-get(){
   local dir=$(nature-dir) &&  mkdir -p $dir && cd $dir

   local url=http://rastergrid.com/blog/wp-content/uploads/2010/06/nature12_src.zip
   local dst=$(basename $url)

   [ ! -f $dst ] && curl -L -O $url
   [ ! -f nature.h ] && unzip  $dst


}
