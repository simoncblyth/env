# === func-gen- : video/ffmpeg/ffmpeg fgp video/ffmpeg/ffmpeg.bash fgn ffmpeg fgh video/ffmpeg
ffmpeg-src(){      echo video/ffmpeg/ffmpeg.bash ; }
ffmpeg-source(){   echo ${BASH_SOURCE:-$(env-home)/$(ffmpeg-src)} ; }
ffmpeg-vi(){       vi $(ffmpeg-source) ; }
ffmpeg-env(){      elocal- ; }
ffmpeg-usage(){ cat << EOU

ffmpeg : video tools
=======================

* https://ffmpeg.org/download.html
* https://trac.ffmpeg.org/wiki/Slideshow

Overview
---------

Compiling by self rather than getting from distro
appears the preferred approach as liable to want to
configure to use hw accel such as nvenc


* x264 requires minimum nasm-2.13, 
* try update nasm via its repo requires, glibc l
  nasm-2.13.01-0.fc24.x86_64 (nasm)  requires



Compilation
-------------

* https://trac.ffmpeg.org/wiki/CompilationGuide
* https://trac.ffmpeg.org/wiki/CompilationGuide/Centos


SDU GPU manual install
-------------------------

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


HW Acceleration : NVENV
-------------------------

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


x264 : repo nasm not found and too old
-------------------------------------------

::

    [simon@localhost x264]$ PKG_CONFIG_PATH="$HOME/ffmpeg_build/lib/pkgconfig" ./configure --prefix="$HOME/ffmpeg_build" --bindir="$HOME/bin" --enable-static
    Found no assembler
    Minimum version is nasm-2.13
    If you really want to compile without asm, configure with --disable-asm.
    [simon@localhost x264]$ 


    [simon@localhost x264]$ yum info nasm
    Loaded plugins: priorities, refresh-packagekit, security
    Installed Packages
    Name        : nasm
    Arch        : x86_64
    Version     : 2.07
    Release     : 7.el6
    Size        : 1.2 M
    Repo        : installed
    From repo   : sl
    Summary     : A portable x86 assembler which uses Intel-like syntax
    URL         : http://www.nasm.us/
    License     : BSD and LGPLv2+ and GPLv2+
    Description : NASM is the Netwide Assembler, a free portable assembler for the Intel
                : 80x86 microprocessor series, using primarily the traditional Intel
                : instruction mnemonics and syntax.


::

    [root@localhost yum.repos.d]# yum install nasm
    Loaded plugins: priorities, refresh-packagekit, security
    nasm                                                                                                                                                                                 | 3.0 kB     00:00     
    nasm/primary_db                                                                                                                                                                      | 5.1 kB     00:00     
    Setting up Install Process
    Resolving Dependencies
    --> Running transaction check
    ---> Package nasm.x86_64 0:2.07-7.el6 will be updated
    ---> Package nasm.x86_64 0:2.13.01-0.fc24 will be an update
    --> Processing Dependency: libc.so.6(GLIBC_2.14)(64bit) for package: nasm-2.13.01-0.fc24.x86_64
    --> Finished Dependency Resolution
    Error: Package: nasm-2.13.01-0.fc24.x86_64 (nasm)
               Requires: libc.so.6(GLIBC_2.14)(64bit)
     You could try using --skip-broken to work around the problem
    ** Found 19 pre-existing rpmdb problem(s), 'yum check' output follows:
    cyrus-sasl-lib-2.1.23-15.el6_6.2.x86_64 is a duplicate with cyrus-sasl-lib-2.1.23-13.el6_3.1.x86_64
    keyutils-libs-1.4-5.el6.x86_64 is a duplicate with keyutils-libs-1.4-4.el6.x86_64
    krb5-libs-1.10.3-65.el6.x86_64 is a duplicate with krb5-libs-1.10.3-10.el6_4.6.x86_64
    libX11-1.6.4-3.el6.x86_64 is a duplicate with libX11-1.6.3-2.el6.x86_64




* https://developers.redhat.com/blog/2016/02/17/upgrading-the-gnu-c-library-within-red-hat-enterprise-linux/


EOU
}
ffmpeg-dir(){ echo $HOME/ffmpeg_sources ; }
ffmpeg-cd(){  cd $(ffmpeg-dir); }
ffmpeg-get(){
   local dir=$(ffmpeg-dir) &&  mkdir -p $dir && cd $dir

   [ ! -d ffmpeg ] && git clone https://git.ffmpeg.org/ffmpeg.git ffmpeg

}

ffmpeg-preq-get()
{
   ffmpeg-libx264-get
}

# https://trac.ffmpeg.org/wiki/CompilationGuide/Centos

ffmpeg-libx264-get()
{
   ffmpeg-cd
   git clone --depth 1 http://git.videolan.org/git/x264
   cd x264

   PKG_CONFIG_PATH="$HOME/ffmpeg_build/lib/pkgconfig" ./configure --prefix="$HOME/ffmpeg_build" --bindir="$HOME/bin" --enable-static

}


