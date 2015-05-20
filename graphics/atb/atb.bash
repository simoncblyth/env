# === func-gen- : graphics/atb/atb fgp graphics/atb/atb.bash fgn atb fgh graphics/atb
atb-src(){      echo graphics/atb/atb.bash ; }
atb-source(){   echo ${BASH_SOURCE:-$(env-home)/$(atb-src)} ; }
atb-vi(){       vi $(atb-source) ; }
atb-env(){      elocal- ; }
atb-usage(){ cat << EOU

AntTweakBar
=============

Glumpy provides a binding to this, hence checking it out.

* http://anttweakbar.sourceforge.net/doc/tools:anttweakbar:download

#. Hmm nasty, lots of windows binaries inside the distribution.


Community
----------

* http://sourceforge.net/p/anttweakbar/tickets/
* http://sourceforge.net/p/anttweakbar/discussion/AntTweakBar/


Compilation error from const qualifier mismatch
---------------------------------------------------

* http://sourceforge.net/p/anttweakbar/discussion/AntTweakBar/thread/5bfe30ce/

Find by adding two "const"::

    simon:src blyth$ diff LoadOGLCore.h.original LoadOGLCore.h
    149c149
    < ANT_GL_CORE_DECL(void, glMultiDrawElements, (GLenum mode, const GLsizei *count, GLenum type, const GLvoid* *indices, GLsizei primcount))
    ---
    > ANT_GL_CORE_DECL(void, glMultiDrawElements, (GLenum mode, const GLsizei *count, GLenum type, const GLvoid* const *indices, GLsizei primcount))
    214c214
    < ANT_GL_CORE_DECL(void, glShaderSource, (GLuint shader, GLsizei count, const GLchar* *string, const GLint *length))
    ---
    > ANT_GL_CORE_DECL(void, glShaderSource, (GLuint shader, GLsizei count, const GLchar* const *string, const GLint *length))
    simon:src blyth$ 


Lookahead at issues
----------------------

* http://sourceforge.net/p/anttweakbar/tickets/14/

::

    gl_dyld = dlopen("OpenGL",RTLD_LAZY);
    gl_dyld = dlopen("/System/Library/Frameworks/OpenGL.framework/OpenGL",RTLD_LAZY);





EOU
}
atb-dir(){ echo $(local-base)/env/graphics/atb/AntTweakBar ; }
atb-cd(){  cd $(atb-dir)/$1 ; }
atb-mate(){ mate $(atb-dir) ; }
atb-get(){
   local dir=$(dirname $(atb-dir)) &&  mkdir -p $dir && cd $dir

   local url=http://downloads.sourceforge.net/project/anttweakbar/AntTweakBar_116.zip
   local zip=$(basename $url)

   [ ! -f "$zip" ] && curl -L -O $url
   [ ! -d AntTweakBar ] && unzip $zip

}

atb-make(){

   atb-cd src

   make -f Makefile.osx

}


