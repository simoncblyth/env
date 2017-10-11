# === func-gen- : graphics/nvidia/capturesdk fgp graphics/nvidia/capturesdk.bash fgn capturesdk fgh graphics/nvidia
capturesdk-src(){      echo graphics/nvidia/capturesdk.bash ; }
capturesdk-source(){   echo ${BASH_SOURCE:-$(env-home)/$(capturesdk-src)} ; }
capturesdk-vi(){       vi $(capturesdk-source) ; }
capturesdk-env(){      elocal- ; }
capturesdk-usage(){ cat << EOU

NVIDIA Capture SDK (formerly Grid SDK)
=========================================

* https://developer.nvidia.com/capture-sdk
* http://on-demand.gputechconf.com/gtc/2016/presentation/s6307-shounak-deshpande-get-to-know-the-nvidia-grid-sdk.pdf


To run Capture SDK, please obtain a compatible NVIDIA Driver for your hardware.

For Linux, 384.59 or newer drivers are required for NVIDIA Capture SDK 6.1.
For Windows, 385.05 or newer drivers are required for NVIDIA Capture SDK 6.1 and can be obtained directly from http://www.nvidia.com/drivers.



NVFBC (frame buffer capture)
   brute force capture all on screen, orthogonal to graphics APIs

NVIFR (in-band frame render)
   no-frills RenderTarget capture, supports OpenGL 
   NVIFRToSys
   NVIFRToHWEnc


* https://superuser.com/questions/1190385/nvidia-grid-sdk-capture-sdk


Open Broadcaster
------------------

* https://obsproject.com





EOU
}
capturesdk-dir(){ echo $(local-base)/env/graphics/nvidia/graphics/nvidia-capturesdk ; }
capturesdk-cd(){  cd $(capturesdk-dir); }
capturesdk-mate(){ mate $(capturesdk-dir) ; }
capturesdk-get(){
   local dir=$(dirname $(capturesdk-dir)) &&  mkdir -p $dir && cd $dir

}
