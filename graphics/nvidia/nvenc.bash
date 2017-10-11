# === func-gen- : graphics/nvidia/nvenc fgp graphics/nvidia/nvenc.bash fgn nvenc fgh graphics/nvidia
nvenc-src(){      echo graphics/nvidia/nvenc.bash ; }
nvenc-source(){   echo ${BASH_SOURCE:-$(env-home)/$(nvenc-src)} ; }
nvenc-vi(){       vi $(nvenc-source) ; }
nvenc-env(){      elocal- ; }
nvenc-usage(){ cat << EOU


NVENV : NVIDIA GPU H.264/265 video encoding
================================================  

* perhaps should use higher level : Capture SDK (formerly GRID SDK) 
* https://developer.nvidia.com/capture-sdk


* https://developer.nvidia.com/nvidia-video-codec-sdk
* https://developer.nvidia.com/video-encode-decode-gpu-support-matrix


NVIDIA Linux display driver 378.13 or newer
NVIDIA Windows display driver 378.66 or newer 

* https://en.wikipedia.org/wiki/Nvidia_NVENC
* https://www.bandicam.com/support/tips/nvidia-nvenc/

::

    Nvidia NVENC H264 encoder 
    Nvidia NVENC H265 encoder 


* FFmpeg has supported NVENC since 2014,[8] and is supported in Nvidia drivers.[9]

* http://ffmpeg.org/doxygen/trunk/nvenc_8c_source.html

::

      114 
      115 static void nvenc_print_driver_requirement(AVCodecContext *avctx, int level)
      116 {
      117 #if defined(_WIN32) || defined(__CYGWIN__)
      118     const char *minver = "378.66";
      119 #else
      120     const char *minver = "378.13";
      121 #endif
      122     av_log(avctx, level, "The minimum required Nvidia driver for nvenc is %s or newer\n", minver);
      123 }



EOU
}
nvenc-dir(){ echo $(local-base)/env/graphics/nvidia/graphics/nvidia-nvenc ; }
nvenc-cd(){  cd $(nvenc-dir); }
nvenc-mate(){ mate $(nvenc-dir) ; }
nvenc-get(){
   local dir=$(dirname $(nvenc-dir)) &&  mkdir -p $dir && cd $dir

}
