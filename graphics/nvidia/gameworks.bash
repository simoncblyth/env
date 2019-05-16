# === func-gen- : graphics/nvidia/gameworks fgp graphics/nvidia/gameworks.bash fgn gameworks fgh graphics/nvidia src base/func.bash
gameworks-source(){   echo ${BASH_SOURCE} ; }
gameworks-edir(){ echo $(dirname $(gameworks-source)) ; }
gameworks-ecd(){  cd $(gameworks-edir); }
gameworks-dir(){  echo $LOCAL_BASE/env/graphics/nvidia/GraphicsSamples ; }
gameworks-cd(){   cd $(gameworks-dir); }
gameworks-vi(){   vi $(gameworks-source) ; }
gameworks-env(){  elocal- ; }
gameworks-usage(){ cat << EOU

NVIDIA GameWorks Graphics Samples
====================================

The GameWorks Graphics Samples pack is a resource for cross-platform Vulkan 1.0
(VK10), OpenGL 4 (GL4) and OpenGL ES 2 and 3 (ES2 and ES3) development,
targeting Android, Windows, and Linux (x86/x64 and Linux for Tegra). The
samples run on all four target platforms from a single source base.

FUNCTIONS
----------

gameworks-doc
    open local html documentation



EOU
}
gameworks-get(){
   local dir=$(dirname $(gameworks-dir)) &&  mkdir -p $dir && cd $dir

   [ ! -d GraphicsSamples ] && git clone https://github.com/NVIDIAGameWorks/GraphicsSamples

}


gameworks-doc(){ open $(gameworks-dir)/doc/index.html ; }
