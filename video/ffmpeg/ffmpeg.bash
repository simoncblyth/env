# === func-gen- : video/ffmpeg/ffmpeg fgp video/ffmpeg/ffmpeg.bash fgn ffmpeg fgh video/ffmpeg
ffmpeg-src(){      echo video/ffmpeg/ffmpeg.bash ; }
ffmpeg-source(){   echo ${BASH_SOURCE:-$(env-home)/$(ffmpeg-src)} ; }
ffmpeg-vi(){       vi $(ffmpeg-source) ; }
ffmpeg-env(){      elocal- ; }
ffmpeg-usage(){ cat << EOU

ffmpeg : video tools
=======================

See also
--------

* nvenc-
* nasm-
* x264-


Refs
-----

* https://ffmpeg.org/download.html
* https://trac.ffmpeg.org/wiki/Slideshow
* https://trac.ffmpeg.org/wiki/CompilationGuide/Centos

* https://en.wikibooks.org/wiki/FFMPEG_An_Intermediate_Guide/image_sequence

Overview (2016)
------------------

Compiling by self rather than getting from distro
appears the preferred approach as liable to want to
configure to use hw accel such as nvenc

* x264 requires minimum nasm-2.13, 
* trying to update nasm via its repo dependency fails for newer glibc, so build nasm from source with nasm-


Compilation
-------------

* https://trac.ffmpeg.org/wiki/CompilationGuide
* https://trac.ffmpeg.org/wiki/CompilationGuide/Centos


IHEP GPU Workstation (2018-7-18)
----------------------------------

Following https://trac.ffmpeg.org/wiki/CompilationGuide/Centos

::

   yum info autoconf automake bzip2 cmake freetype-devel gcc gcc-c++ git libtool make mercurial pkgconfig zlib-devel

   sudo yum install autoconf
   sudo yum install automake

   # bzip2  1.0.6 already installed
   # cmake3 3.11.2 already installed

   sudo yum install freetype-devel

   # gcc 4.8.5 already installed
   # gcc-c++ 4.8.5 already installed
   # git 1.8.3.1 already installed

   sudo yum install libtool

   # make 3.82 already installed
   # mercurial 2.6.2 already installed
   # pkgconfig 0.27.1 already installed (from anaconda)
   # zlib-devel 1.2.7 already installed


::

    yum info nasm #  2.10.07 is too old for x264, so install from source with nasm-



SDU GPU manual install (2016-?)
-------------------------------------

Following https://trac.ffmpeg.org/wiki/CompilationGuide/Centos

::

    yum info autoconf automake bzip2 cmake freetype-devel gcc gcc-c++ git libtool make mercurial nasm pkgconfig zlib-devel
 

The below are listed as available::

    cmake    # repo cmake is an old version, a more recent one is already installed
    freetype-devel   # already there 
    zlib-devel       # already there
    libtool          # installed 
    nasm 


Linux Distros
---------------

* https://rpmfusion.org
* https://chrisjean.com/install-ffmpeg-and-ffmpeg-php-on-centos-easily/
* https://linoxide.com/linux-how-to/install-ffmpeg-centos-7/

Subtitles
-----------

* https://trac.ffmpeg.org/wiki/HowToBurnSubtitlesIntoVideo



ffmpeg commandline commands/options
---------------------------------------

::

  ffmpeg -buildconf


Selection of options
~~~~~~~~~~~~~~~~~~~~~~~

* http://hamelot.io/visualization/using-ffmpeg-to-convert-a-set-of-images-into-a-video/


H.264
--------

* https://trac.ffmpeg.org/wiki/Encode/H.264


Video from images
--------------------

* https://superuser.com/questions/624567/how-to-create-a-video-from-images-using-ffmpeg


HW Acceleration : NVENC
-------------------------

* see nvenc-

* https://trac.ffmpeg.org/wiki/HWAccelIntro

NVENC is an API developed by NVIDIA which enables the use of NVIDIA GPU cards
to perform H.264 and HEVC encoding. FFmpeg supports NVENC through the
h264_nvenc and hevc_nvenc encoders. In order to enable it in FFmpeg you need:

* A supported GPU
* Supported drivers
* ffmpeg configured without --disable-nvenc

Download the NVIDIA Video Codec SDK, and check the website for more info on the supported GPUs and drivers.

Usage example::

    ffmpeg   -i input \
             -c:v h264_nvenc \         # setting video codec 
             -profile high444p \       # 4:4:4 is without chroma subsampling, likely to be large file and compatibility issues with playing ? 
             -pixel_format yuv444p \
             -preset default \
             output.mp4

You can see available presets, other options, and encoder info with::

    ffmpeg -h encoder=h264_nvenc 
    ffmpeg -h encoder=hevc_nvenc

Note: If you get the No NVENC capable devices found error make sure you're 
encoding to a supported pixel format. See encoder info as shown above.



NVIDIA VIDEO CODEC SDK
---------------------------

NVIDIA directs most users to FFmpeg to use the SDK at higher level 

* https://developer.nvidia.com/nvidia-video-codec-sdk
* https://developer.nvidia.com/FFmpeg

You can now use FFMPEG to accelerate video encoding and decoding using NVENC
and NVDEC, respectively.



HEVC
-----

* https://en.wikipedia.org/wiki/High_Efficiency_Video_Coding

also known as H.265 and MPEG-H Part 2, is a video compression standard, one of
several potential successors to the widely used AVC (H.264 or MPEG-4 Part 10).
In comparison to AVC, HEVC offers about double the data compression ratio at
the same level of video quality, or substantially improved video quality at the
same bit rate. It supports resolutions up to 8192×4320, including 8K UHD.



On September 19, 2017, Apple released iOS 11 and tvOS 11 with HEVC encoding & decoding support.[89][83]

* https://developer.apple.com/videos/play/wwdc2017/503/

On September 25, 2017, Apple released macOS High Sierra with HEVC encoding & decoding support.
On September 28, 2017, GoPro released the Hero6 Black action camera, with 4K60P HEVC video encoding.[90]


HEVC Competitor : AV1
-----------------------

* https://en.wikipedia.org/wiki/AOMedia_Video_1
* http://www.streamingmedia.com/Articles/Editorial/Featured-Articles/The-State-of-Video-Codecs-2016-110117.aspx


4:2:2 vs 4:4:4 vs 4:2:0 : Chroma subsampling
-----------------------------------------------

* http://www.rtings.com/tv/learn/chroma-subsampling

chroma subsampling reduces the amount of color information in the signal
to allow more luminance data instead. This allows you to maintain picture
clarity while effectively reducing the file size up to 50%

* 4:4:4 no subsampling, no compression


Brainiarc7/ffmpeg-hevc-encode-nvenc.md
-----------------------------------------

Encoding high-quality HEVC content with FFmpeg - based NVENC encoder on supported hardware:


* https://gist.github.com/Brainiarc7/8b471ff91319483cdb725f615908286e



Making mp4 from ppm snaps
---------------------------

See below functions for example of making a mp4 H.264 video from ppm snaps using ffmpeg 

::

    okop-
    okop-snap
    okop-snap-mp4



EOU
}
ffmpeg-dir(){ echo $(local-base)/env/video/ffmpeg/ffmpeg ; }
ffmpeg-cd(){  cd $(ffmpeg-dir); }
ffmpeg-get(){
   local dir=$(dirname $(ffmpeg-dir)) &&  mkdir -p $dir && cd $dir

   [ ! -d ffmpeg ] && git clone https://git.ffmpeg.org/ffmpeg.git ffmpeg
}

ffmpeg-info(){ cat << EOI
 
   ffmpeg-dir    : $(ffmpeg-dir)
   ffmpeg-prefix : $(ffmpeg-prefix)
   uname -a      : $(uname -a)
   date          : $(date) 

EOI
}


ffmpeg-preqs()
{
   nasm-
   nasm--   # recent nasm required by x264
   
   x264-
   x264--
}


ffmpeg-prefix(){ echo $(local-base)/env  ; }

ffmpeg-configure-notes(){ cat << EON

https://trac.ffmpeg.org/wiki/CompilationGuide/Centos#x264

x264 : Requires ffmpeg to be configured with --enable-gpl --enable-libx264 


Note that the configure appears to hang for several minutes 
before producing any output, before dumping 298 lines to stdout 

/home/blyth/local/env/video/ffmpeg/ffmpeg/ffmpeg-configure.log

::

    [blyth@localhost ffmpeg]$ wc -l ffmpeg-configure.log
    298 ffmpeg-configure.log

    [blyth@localhost ffmpeg]$ head -20 ffmpeg-configure.log
    [blyth@localhost ffmpeg]$ ffmpeg-configure
    install prefix            /home/blyth/local/env
    source path               .
    C compiler                gcc
    C library                 glibc
    ARCH                      x86 (generic)
    big-endian                no
    runtime cpu detection     yes
    standalone assembly       yes
    x86 assembler             nasm
    MMX enabled               yes
    MMXEXT enabled            yes
    3DNow! enabled            yes
    3DNow! extended enabled   yes
    SSE enabled               yes
    SSSE3 enabled             yes
    AESNI enabled             yes
    AVX enabled               yes
    AVX2 enabled              yes
    AVX-512 enabled           yes


ffmpeg-configure-0
    worked to build ffmpeg which succeeded to make an mp4 from some ppm
    but fails to link in obs- lots of missing symbols


Observations : this is an old school dirty configure build, not 
a CMake separated bdir from sdir. So on changing configure options 
have to blow away the distribution and get it again, to be sure do not have 
a mixed configuration.

EON
}

ffmpeg-configure-0()
{
    ffmpeg-cd
    echo $FUNCNAME : this configure takes some time before any output appears... 
    date

    PKG_CONFIG_PATH="$(ffmpeg-prefix)/lib/pkgconfig" ./configure \
           --prefix="$(ffmpeg-prefix)" \
           --extra-cflags="-I$(ffmpeg-prefix)/include" \
           --extra-ldflags="-L$(ffmpeg-prefix)/lib -ldl" \
           --pkg-config-flags="--static" \
           --enable-gpl \
           --enable-libx264 

    date
}




ffmpeg-configure()
{
    ffmpeg-cd
    echo $FUNCNAME : this configure takes some time before any output appears... 
    date

    PKG_CONFIG_PATH="$(ffmpeg-prefix)/lib/pkgconfig" ./configure \
           --pkg-config-flags="--static" \
           --prefix="$(ffmpeg-prefix)" \
           --extra-cflags="-I$(ffmpeg-prefix)/include" \
           --extra-ldflags="-L$(ffmpeg-prefix)/lib -ldl" \
           --extra-libs=-lpthread \
           --extra-libs=-lm \
           --enable-pic \
           --enable-shared \
           --enable-gpl \
           --enable-libx264 

    date
}

ffmpeg-configure-scratch(){ cat << EOS

    libavutil/x86/float_dsp.o: relocation R_X86_64_32S against .rodata can not be used when making a shared object; recompile with -fPIC

    --enable-libfdk_aac \
    --enable-libfreetype \
    --enable-libmp3lame \
    --enable-libopus \
    --enable-libvorbis \
    --enable-libvpx \
    --enable-libx264 \
    --enable-libx265 \
    --enable-nonfree

    https://trac.ffmpeg.org/ticket/2784

EOS
}

ffmpeg-install()
{
    ffmpeg-cd

    make
    make install
}

ffmpeg--()
{
    ffmpeg-get
    ffmpeg-configure
    ffmpeg-install
}

ffmpeg-export()
{
   export LD_LIBRARY_PATH=$(ffmpeg-prefix)/lib:$LD_LIBRARY_PATH
}
