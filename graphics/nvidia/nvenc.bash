# === func-gen- : graphics/nvidia/nvenc fgp graphics/nvidia/nvenc.bash fgn nvenc fgh graphics/nvidia
nvenc-src(){      echo graphics/nvidia/nvenc.bash ; }
nvenc-source(){   echo ${BASH_SOURCE:-$(env-home)/$(nvenc-src)} ; }
nvenc-vi(){       vi $(nvenc-source) ; }
nvenc-env(){      elocal- ; }
nvenc-usage(){ cat << EOU


NVENC : NVIDIA GPU H.264/265 video encoding
================================================  

* see ffmpeg-


NvPipe : easier way of using nvenc ?
--------------------------------------

* https://developer.nvidia.com/gtc/2019/video/S9490/video

  Integrating nvenc on V100 with OptiX and remote web browser 

  * V100 have 3 NVENC chips, allowing H264 H265 encoding of video streams
  * low latency visualization streaming 
  * Broadway.js in browser

  * p17 : NvPipe_CreateEncoder  / NvPipe_Encode 


Informative NvPipe issue
-----------------------------

* https://github.com/NVIDIA/NvPipe/issues/68

Hey, thanks for the positive feedback!

Broadway.js has been successfully used with NvPipe in the past, e.g., in
Benjamín Hernández's SIGHT project: https://github.com/benjha/Sight_FrameServer
However, as a pure software decoder, Broadway.js can not leverage all hardware
acceleration capabilities.

You can also manually wrap the produced H.264 bitstream in an MP4 container,
and stream the wrapped packages via WebSockets to a JavaScript client that uses
Media Source Extensions to feed the data into a standard HTML video tag. The
browser can then use hardware-accelerated decoding under the hood.

To produce a valid MP4 stream that can be consumed by the browser, you first
need to produce the overall MP4 header which contains some info on the video
stream such as codec, resolution, frame rate etc., and then you wrap each H.264
frame behind a small MP4 per-frame header. I suggest checking the MP4
specification.

Note that you need to implement some flow control on the server side, as the
browser will try to play the stream at the constant specified frame rate. Just
make sure that the server doesn't render/send frames faster, otherwise the
browser will start buffering. On the contrary, try to stay close to a buffer
underrun at all times.

We have a working internal prototype of this approach. Please feel free to
reach out directly via tbiedert@nvidia.com to discuss how you can leverage this
for your usecase.


NVENC Links
---------------

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
