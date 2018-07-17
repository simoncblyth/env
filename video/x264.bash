# === func-gen- : video/x264 fgp video/x264.bash fgn x264 fgh video
x264-src(){      echo video/x264.bash ; }
x264-source(){   echo ${BASH_SOURCE:-$(env-home)/$(x264-src)} ; }
x264-vi(){       vi $(x264-source) ; }
x264-usage(){ cat << EOU

X264
======

* see ffmpeg- nasm-

x264 is a free software library and application for encoding video streams into
the H.264/MPEG-4 AVC compression format, and is released under the terms of the
GNU GPL

https://www.videolan.org/developers/x264.html

::

    simon:x264 blyth$ nasm -v
    NASM version 0.98.40 (Apple Computer, Inc. build 11) compiled on Feb 24 2015
    simon:x264 blyth$ PATH=$LOCAL_BASE/env/bin:$PATH which nasm
    /usr/local/env/bin/nasm

    simon:x264 blyth$ PATH=$LOCAL_BASE/env/bin:$PATH nasm -v
    NASM version 2.13.01 compiled on Oct 11 2017
    simon:x264 blyth$ 


IHEP Linux GPU Workstation (July 2018)
-------------------------------------------

::

    [blyth@localhost ~]$ x264-;x264-info

       x264-dir   : /home/blyth/local/env/video/x264/x264
       which nasm : /home/blyth/local/env/bin/nasm  
       nasm -v    : NASM version 2.13.01 compiled on Jul 17 2018
       uname -a   : Linux localhost.localdomain 3.10.0-862.6.3.el7.x86_64 #1 SMP Tue Jun 26 16:32:21 UTC 2018 x86_64 x86_64 x86_64 GNU/Linux
       date       : Tue Jul 17 15:36:57 CST 2018

::

    [blyth@localhost x264]$ x264-configure
    platform:      X86_64
    byte order:    little-endian
    system:        LINUX
    cli:           yes
    libx264:       internal
    shared:        no
    static:        yes
    asm:           yes
    interlaced:    yes
    avs:           avxsynth
    lavf:          no
    ffms:          no
    mp4:           no
    gpl:           yes
    thread:        posix
    opencl:        yes
    filters:       crop select_every
    lto:           no
    debug:         no
    gprof:         no
    strip:         no
    PIC:           no
    bit depth:     all
    chroma format: all

    You can run 'make' or 'make fprofiled' now.



* note that bit depth has become "all" not "8" 


OSX D
-------

::

    simon:x264 blyth$ x264-build
    platform:      X86_64
    byte order:    little-endian
    system:        MACOSX
    cli:           yes
    libx264:       internal
    shared:        no
    static:        yes
    asm:           yes
    interlaced:    yes
    avs:           avxsynth
    lavf:          no
    ffms:          no
    mp4:           no
    gpl:           yes
    thread:        posix
    opencl:        yes
    filters:       crop select_every 
    lto:           no
    debug:         no
    gprof:         no
    strip:         no
    PIC:           no
    bit depth:     8
    chroma format: all

    You can run 'make' or 'make fprofiled' now.


Linux SG
-----------

::

    [simon@localhost x264]$ x264-configure
    platform:      X86_64
    byte order:    little-endian
    system:        LINUX
    cli:           yes
    libx264:       internal
    shared:        no
    static:        yes
    asm:           yes
    interlaced:    yes
    avs:           avxsynth
    lavf:          no
    ffms:          no
    mp4:           no
    gpl:           yes
    thread:        posix
    opencl:        yes
    filters:       crop select_every 
    lto:           no
    debug:         no
    gprof:         no
    strip:         no
    PIC:           no
    bit depth:     8
    chroma format: all

    You can run 'make' or 'make fprofiled' now.





D::

    simon:x264 blyth$ make install
    install -d /usr/local/env/bin
    install x264 /usr/local/env/bin
    install -d /usr/local/env/include
    install -d /usr/local/env/lib
    install -d /usr/local/env/lib/pkgconfig
    install -m 644 ./x264.h /usr/local/env/include
    install -m 644 x264_config.h /usr/local/env/include
    install -m 644 x264.pc /usr/local/env/lib/pkgconfig
    install -m 644 libx264.a /usr/local/env/lib
    ranlib /usr/local/env/lib/libx264.a


SG::

    [simon@localhost x264]$ make install
    install -d /usr/local/env/bin
    install x264 /usr/local/env/bin
    install -d /usr/local/env/include
    install -d /usr/local/env/lib
    install -d /usr/local/env/lib/pkgconfig
    install -m 644 ./x264.h /usr/local/env/include
    install -m 644 x264_config.h /usr/local/env/include
    install -m 644 x264.pc /usr/local/env/lib/pkgconfig
    install -m 644 libx264.a /usr/local/env/lib
    ranlib /usr/local/env/lib/libx264.a
    [simon@localhost x264]$ 



2018 July Linux IHEP 
----------------------

::

    [blyth@localhost x264]$ x264 
    x264 [error]: No input file. Run x264 --help for a list of options.
    [blyth@localhost x264]$ x264 --help
    x264 core:155
    Syntax: x264 [options] -o outfile infile

    Infile can be raw (in which case resolution is required),
      or YUV4MPEG (*.y4m),
      or Avisynth if compiled with support (yes).
      or libav* formats if compiled with lavf support (no) or ffms support (no).
    Outfile type is selected by filename:
     .264 -> Raw bytestream
     .mkv -> Matroska
     .flv -> Flash Video
     .mp4 -> MP4 if compiled with GPAC or L-SMASH support (no)
    Output bit depth: 8/10
    .
    Options:

      -h, --help                  List basic options
          --longhelp              List more options
          --fullhelp              List all options

    Example usage:

          Constant quality mode:
                x264 --crf 24 -o <output> <input>
    ...



EOU
}
x264-dir(){ echo $(local-base)/env/video/x264/x264 ; }
x264-cd(){  cd $(x264-dir); }
x264-get(){
   local dir=$(dirname $(x264-dir)) &&  mkdir -p $dir && cd $dir

   [ ! -d x264 ] && $FUNCNAME-
}

x264-get-(){
   # shallow clone, skips full history 
   git clone --depth 1 http://git.videolan.org/git/x264
}

x264-get-workaround-()
{
   #git clone --depth 1 ssh://blyth@simon.phys.ntu.edu.tw/usr/local/env/video/x264/x264
   git clone blyth@simon.phys.ntu.edu.tw:x264   #   see git- 
}


x264-env(){      elocal- ; nasm- ; }   ## need recent nasm in PATH

x264-configure()
{
   x264-cd
   ./configure --prefix="$(local-base)/env" --enable-static

   ##  https://trac.ffmpeg.org/wiki/CompilationGuide/Centos
   ##  PKG_CONFIG_PATH="$HOME/ffmpeg_build/lib/pkgconfig" ./configure --prefix="$HOME/ffmpeg_build" --bindir="$HOME/bin" --enable-static
}

x264-info(){ cat << EOI

   x264-dir   : $(x264-dir)
   which nasm : $(which nasm)  
   nasm -v    : $(nasm -v)
   uname -a   : $(uname -a)
   date       : $(date)

EOI
}

x264--()
{
   x264-get
   x264-configure
   make
   make install
}


