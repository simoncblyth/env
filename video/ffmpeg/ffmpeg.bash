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
# https://trac.ffmpeg.org/wiki/CompilationGuide/Centos

Overview
---------

Compiling by self rather than getting from distro
appears the preferred approach as liable to want to
configure to use hw accel such as nvenc


* x264 requires minimum nasm-2.13, 
* trying to update nasm via its repo dependency fails for newer glibc, so build nasm from source


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


EOU
}
ffmpeg-dir(){ echo $HOME/ffmpeg_sources ; }
ffmpeg-cd(){  cd $(ffmpeg-dir); }
ffmpeg-get(){
   local dir=$(ffmpeg-dir) &&  mkdir -p $dir && cd $dir

   [ ! -d ffmpeg ] && git clone https://git.ffmpeg.org/ffmpeg.git ffmpeg

}

ffmpeg-preqs()
{
   nasm-
   nasm--
   
   x264-
   x264--
}





