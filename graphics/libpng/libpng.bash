# === func-gen- : graphics/libpng/libpng fgp graphics/libpng/libpng.bash fgn libpng fgh graphics/libpng
libpng-src(){      echo graphics/libpng/libpng.bash ; }
libpng-source(){   echo ${BASH_SOURCE:-$(env-home)/$(libpng-src)} ; }
libpng-vi(){       vi $(libpng-source) ; }
libpng-env(){      elocal- ; }
libpng-usage(){ cat << EOU

libpng usage example 
========================

* http://www.libpng.org/pub/png/book/sources.html
* http://netpbm.sourceforge.net/doc/ppm.html


PPM is a trivial image format that can be 
created by raytrace- with no OpenGL involved.  
This allows to create a render on a headless
node purely computationally 
without the complexity of getting OpenGL to 
work.

Usage example
--------------

::

    cd ~/simoncblyth.bitbucket.io/env/presentation/j1808

    epsilon:j1808 blyth$ libpng-
    epsilon:j1808 blyth$ libpng-- j1808_top_rtx.ppm
    wpng-ing ppm j1808_top_rtx.ppm into png j1808_top_rtx.png
    Encoding image data...
    Done.
    epsilon:j1808 blyth$ libpng-- j1808_top_ogl.ppm
    wpng-ing ppm j1808_top_ogl.ppm into png j1808_top_ogl.png
    Encoding image data...
    Done.

    epsilon:j1808 blyth$ du -h *
    2.1M	j1808_top_ogl.png
    5.9M	j1808_top_ogl.ppm
    640K	j1808_top_rtx.png
    5.9M	j1808_top_rtx.ppm
    epsilon:j1808 blyth$ 




EOU
}

libpng-dir(){ echo $(local-base)/env/graphics/libpng/pngbook-20080316-src ; }
libpng-bdir(){ echo $(libpng-dir)/build ; }
libpng-cd(){  cd $(libpng-dir); }
libpng-bcd(){  cd $(libpng-bdir); }
libpng-get(){
   local dir=$(dirname $(libpng-dir)) &&  mkdir -p $dir && cd $dir
   local url=http://prdownloads.sourceforge.net/png-mng/pngbook-20080316-src.tar.gz
   local tgz=$(basename $url)
   local nam=${tgz/.tar.gz}

   [ ! -f "$tgz" ] && curl -L -O $url
   [ ! -d "$nam" ] && tar zxvf $tgz    
}

libpng-make(){
   local bdir=$(libpng-bdir)
   mkdir -p $bdir
   libpng-bcd

   cp $(libpng-dir)/wpng.c .
   cp $(libpng-dir)/writepng.c .
   cp $(libpng-dir)/writepng.h .

   perl -pi -e 's,#include "png.h",#include "png.h"\n#include "zlib.h"\n,' writepng.c 

   make -f $(libpng-dir)/Makefile.unx wpng  $*
}


libpng-make-osx(){
   libpng-make  CC=clang LD=clang \
        PNGDIR="/opt/X11/lib"  \
       PNGLIBd="/opt/X11/lib/libpng.dylib" \
        PNGINC="-I/opt/X11/include/libpng15" \
          XINC="-I/opt/X11/include" \
          XLIB="-L/opt/X11/lib -lX11" \
          ZLIBd="/usr/lib/libz.dylib " \
          ZINC="-I/usr/include" 
}

libpng-wpng(){
  $(libpng-bdir)/wpng $*
}

libpng-test-osx(){
   cat /System/Library/Frameworks/Tk.framework/Versions/8.5/Resources/Scripts/demos/images/teapot.ppm | libpng-wpng > teapot.png
}

libpng--(){
  local ppm=${1:-out.ppm}
  local png=${ppm/.ppm}.png 
  echo $msg wpng-ing ppm $ppm into png $png  
  case $ppm in 
    *ppm) cat $ppm | libpng-wpng > $png && open $png ;;
       *) echo expecting path ending .ppm ;; 
  esac
}


