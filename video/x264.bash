# === func-gen- : video/x264 fgp video/x264.bash fgn x264 fgh video
x264-src(){      echo video/x264.bash ; }
x264-source(){   echo ${BASH_SOURCE:-$(env-home)/$(x264-src)} ; }
x264-vi(){       vi $(x264-source) ; }
x264-env(){      elocal- ; }
x264-usage(){ cat << EOU

X264
======

x264 is a free software library and application for encoding video streams into
the H.264/MPEG-4 AVC compression format, and is released under the terms of the
GNU GPL

https://www.videolan.org/developers/x264.html


EOU
}
x264-dir(){ echo $(local-base)/env/video/x264/x264 ; }
x264-cd(){  cd $(x264-dir); }
x264-get(){
   local dir=$(dirname $(x264-dir)) &&  mkdir -p $dir && cd $dir

   [ ! -d x264 ] && git clone --depth 1 http://git.videolan.org/git/x264

    # PKG_CONFIG_PATH="$HOME/ffmpeg_build/lib/pkgconfig" ./configure --prefix="$HOME/ffmpeg_build" --bindir="$HOME/bin" --enable-static

}
