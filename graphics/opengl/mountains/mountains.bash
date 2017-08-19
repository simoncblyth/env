# === func-gen- : graphics/opengl/mountains/mountains fgp graphics/opengl/mountains/mountains.bash fgn mountains fgh graphics/opengl/mountains
mountains-src(){      echo graphics/opengl/mountains/mountains.bash ; }
mountains-source(){   echo ${BASH_SOURCE:-$(env-home)/$(mountains-src)} ; }
mountains-vi(){       vi $(mountains-source) ; }
mountains-env(){      elocal- ; }
mountains-usage(){ cat << EOU

Mountains Demo
=================

* http://rastergrid.com/blog/2010/10/gpu-based-dynamic-geometry-lod/
* http://rastergrid.com/blog/downloads/mountains-demo/


Issues
--------

* timing, navigation
* retina in quad corner problem when press SPACE

Instance Culling via Geometry Shader
---------------------------------------

* http://rastergrid.com/blog/2010/02/instance-culling-using-geometry-shaders/

Examples
----------

* https://github.com/nvpro-samples/gl_occlusion_culling


Start Converting to GLFW/GLEQ in bitbucket
---------------------------------------------

* bitbucket.org/simoncblyth/mountains


texture2d -> texture
-----------------------

* https://stackoverflow.com/questions/26266198/glsl-invalid-call-of-undeclared-identifier-texture2d

Cripes. Finally found the answer right after I posted the question. texture2D has been replaced by texture.

Yes, be aware that on OS X #version 150 can only mean #version 150 core. On
other platforms where compatibility profiles are implemented, you can continue
to use things that were deprecated beginning in GLSL 1.30 such as texture2D if
you write #version 150 compatibility. You really don't want that, but it's
worth mentioning ;) – Andon M. Coleman




EOU
}


mountains-sdir(){ echo $HOME/mountains ; }
mountains-bdir(){ echo $(local-base)/env/graphics/opengl/mountains.build ; }

mountains-cd(){  cd $(mountains-sdir); }
mountains-c(){  cd $(mountains-sdir); }
mountains-bcd(){  cd $(mountains-bdir); }


mountains-get-original()
{
   local url=http://www.rastergrid.com/blog/wp-content/uploads/2010/10/mountains-src.zip
   local dst=$(basename $url)
   [ ! -f "$dst" ] && curl -L -O $url
   [ ! -d shaders ] && unzip $dst
}

mountains-get-resources()
{
   # these now in 
   local url2=http://www.rastergrid.com/blog/wp-content/uploads/2010/10/mountains-win32.zip
   local dst2=$(basename $url2)
   [ ! -f "$dst2" ] && curl -L -O $url2

   [ ! -d models   ] && unzip $dst2 'models/*' 
   [ ! -d textures ] && unzip $dst2 'textures/*' 
}


mountains-get(){
   #local dir=$(dirname $(mountains-dir)) &&  mkdir -p $dir && cd $dir
   cd $HOME

   [ ! -d mountains ] && hg clone ssh://hg@bitbucket.org/simoncblyth/mountains
   
}

mountains-name(){ echo Mountains ; }

mountains-wipe(){
   local bdir=$(mountains-bdir)
   rm -rf $bdir
}

mountains-cmake(){
   local iwd=$PWD

   local bdir=$(mountains-bdir)
   mkdir -p $bdir
 
   opticks- 
 
   mountains-bcd 
   cmake \
       -DCMAKE_BUILD_TYPE=Debug \
       -DCMAKE_INSTALL_PREFIX=$(opticks-prefix) \
       $(mountains-sdir)

   cd $iwd
}

mountains-make(){
   local iwd=$PWD

   mountains-bcd
   make $*
   cd $iwd
}

mountains-install(){
   mountains-make install
}

mountains--()
{
    mountains-wipe
    mountains-cmake
    mountains-make
    mountains-install
}

