# === func-gen- : graphics/opengl/mountains/mountains fgp graphics/opengl/mountains/mountains.bash fgn mountains fgh graphics/opengl/mountains
mountains-src(){      echo graphics/opengl/mountains/mountains.bash ; }
mountains-source(){   echo ${BASH_SOURCE:-$(env-home)/$(mountains-src)} ; }
mountains-vi(){       vi $(mountains-source) ; }
mountains-env(){      elocal- ; }
mountains-usage(){ cat << EOU


http://rastergrid.com/blog/2010/10/gpu-based-dynamic-geometry-lod/
http://rastergrid.com/blog/downloads/mountains-demo/



EOU
}
mountains-dir(){ echo $(local-base)/env/graphics/opengl/mountains/shaders ; }
mountains-cd(){  cd $(mountains-dir); }
mountains-c(){  cd $(mountains-dir); }

mountains-get(){
   local dir=$(dirname $(mountains-dir)) &&  mkdir -p $dir && cd $dir

   local url=http://www.rastergrid.com/blog/wp-content/uploads/2010/10/mountains-src.zip
   local dst=$(basename $url)
   [ ! -f "$dst" ] && curl -L -O $url
   [ ! -d shaders ] && unzip $dst


}
