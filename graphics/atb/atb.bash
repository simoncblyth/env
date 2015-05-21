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

* https://github.com/bagage/AntTweakBar


Alternative OpenGL GUIs that work with GLFW3 ?
------------------------------------------------

Seems AntTweakBar is abandoned... so look for alternates:

* :google:`AntTweakBar alternatives`
* http://gamedev.stackexchange.com/questions/3617/good-gui-for-opengl

librocket
~~~~~~~~~~~

* http://librocket.com
* https://github.com/libRocket/libRocket



* python scripting integration, using boost python

ImGUI (MIT) : Immediate Mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://github.com/ocornut/imgui
* includes an embedded console : developer-centric


Summarizing an article on IMGUI
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* http://www.johno.se/book/imgui.html

GUIs traditionally duplicate some portion of application state 
and demand a synchronization so that state sloshes back and forth
between GUI and application.  

IMGUI eliminates the syncing by always passing the state...

* widgets no longer objects, become procedural method calls

* simplicity comes at expense of constantly resubmitting state 
  and redrawing the user interface at real-time rates. 



CEGUI (MIT)
~~~~~~~~~~~~~

Cross-platform XML based GUI system

* http://cegui.org.uk/wiki/Main_Page
* https://bitbucket.org/cegui/cegui
* http://cegui.org.uk
* http://static.cegui.org.uk/docs/current/


GWEN : GUI Without Extravagant Nonsense  (like documentation)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://github.com/garrynewman/GWEN

GIGI
~~~~~

* http://gigi.sourceforge.net




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


Uncomment TwSimpleGLFW example and try to build with Makefile.osx
--------------------------------------------------------------------

Undefined symbolds, probably due to GLFW update to GLFW3::

    Undefined symbols for architecture x86_64:
      "_glfwEnable", referenced from:
          _main in TwSimpleGLFW-e4d333.o
      "_glfwGetDesktopMode", referenced from:
          _main in TwSimpleGLFW-e4d333.o
      "_glfwGetWindowParam", referenced from:
          _main in TwSimpleGLFW-e4d333.o
      "_glfwOpenWindow", referenced from:
          _main in TwSimpleGLFW-e4d333.o
      "_glfwSetMousePosCallback", referenced from:
          _main in TwSimpleGLFW-e4d333.o
      "_glfwSetMouseWheelCallback", referenced from:
          _main in TwSimpleGLFW-e4d333.o


* http://sourceforge.net/p/anttweakbar/tickets/11/
* http://sourceforge.net/p/anttweakbar/tickets/_discuss/thread/3b834fc3/5a6e/attachment/0001-Add-GLFW3-integration.patch


GLFW3 Patch
------------

::

    simon:AntTweakBar blyth$ git apply 0001-Add-GLFW3-integration.patch 
    0001-Add-GLFW3-integration.patch:26: trailing whitespace.
    // For GLFW3 event callbacks
    0001-Add-GLFW3-integration.patch:27: trailing whitespace.
    // You should define GLFW_CDECL before including AntTweakBar.h if your version of GLFW uses cdecl calling convensions
    0001-Add-GLFW3-integration.patch:28: trailing whitespace.
    typedef struct GLFWwindow GLFWwindow;
    0001-Add-GLFW3-integration.patch:29: trailing whitespace.
    #ifdef GLFW_CDECL
    0001-Add-GLFW3-integration.patch:30: trailing whitespace.
        TW_API int TW_CDECL_CALL TwEventMouseButtonGLFW3cdecl(GLFWwindow *window, int glfwButton, int glfwAction, int glfwMods);
    error: patch failed: src/Makefile:43
    error: src/Makefile: patch does not apply
    error: patch failed: src/Makefile.osx:40
    error: src/Makefile.osx: patch does not apply
    simon:AntTweakBar blyth$ vi src/Makefile +43
    simon:AntTweakBar blyth$ vi src/Makefile.osx +40
    simon:AntTweakBar blyth$ 

Failed to apply to the TARGET lines in Makefiles::

     39 # name of the application:
     40 TARGET      = AntTweakBar


Suspect need to stuff into git repo for this to work..



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

atb-clean-typs(){ cat << EOT
vcproj
vcproj.filters
vcxproj
vcxproj.filters
dll
exe
lib
sln
EOT
}

atb-clean()
{
   atb-cd .. 
   local typ
   for typ in $(atb-clean-typs) ; do 
       find AntTweakBar -name "*.$typ" -delete 
   done
}

atb-git-init()
{
   atb-cd 
   [ -d .git ] && echo $msg git repo already exists delete and rerun && return 
   git init 
   git add .
   git commit -m "initial commit see atb- $FUNCNAME "
}



atb-make(){

   atb-cd src

   make -f Makefile.osx

}


atb-glfw-patch()
{
   atb-cd

   # http://sourceforge.net/p/anttweakbar/tickets/11/
   local url=http://sourceforge.net/p/anttweakbar/tickets/_discuss/thread/3b834fc3/5a6e/attachment/0001-Add-GLFW3-integration.patch
   local nam=$(basename $url)
   [ ! -f "$nam" ] && curl -L -O $url

}





