# === func-gen- : graphics/nvidia/nvidia fgp graphics/nvidia/nvidia.bash fgn nvidia fgh graphics/nvidia
nvidia-src(){      echo graphics/nvidia/nvidia.bash ; }
nvidia-source(){   echo ${BASH_SOURCE:-$(env-home)/$(nvidia-src)} ; }
nvidia-vi(){       vi $(nvidia-source) ; }
nvidia-env(){      elocal- ; }
nvidia-usage(){ cat << EOU


NVIDIA
========

Graphics on Tesla GPUs 
-------------------------

* http://devblogs.nvidia.com/parallelforall/interactive-supercomputing-in-situ-visualization-tesla-gpus/


Enabling graphics operation on compute GPUs
--------------------------------------------

* https://devtalk.nvidia.com/default/topic/525927/display-driver-failed-installation-with-cuda-5-0/


IHEP hgpu01
------------

::

    -bash-4.1$ which nvidia-smi
    /usr/bin/nvidia-smi
    -bash-4.1$ nvidia-smi --format=csv --query-gpu=gom.current
    FATAL: Module nvidia not found.
    NVIDIA: failed to load the NVIDIA kernel module.
    NVIDIA-SMI has failed because it couldn't communicate with NVIDIA driver. Make sure that latest NVIDIA driver is installed and running.


Flavors of NVIDIA Tesla K20
------------------------------

* https://devtalk.nvidia.com/default/topic/534299/tesla-k20c-or-k20m-/

  * K20c is active cooled, so it can be used in a workstation.
  * K20m is passive cooled, it requires a server chassis. 
    Aside from the cooling option, the specs are the same: 13 SXM,,5GB of memory. 
  * There is also a different passive cooled model, K20x, with more memory (6GB) 
    and higher core count (14 SXM).

NVIDIA On Linux
-----------------

* http://docs.nvidia.com/cuda/cuda-getting-started-guide-for-linux/#axzz3QYdr9xLh


::

    -bash-4.1$ lspci | grep -i nvidia
    03:00.0 3D controller: NVIDIA Corporation GK110GL [Tesla K20m] (rev a1)
    84:00.0 3D controller: NVIDIA Corporation GK110GL [Tesla K20m] (rev a1)
    -bash-4.1$ 





EOU
}
nvidia-dir(){ echo $(local-base)/env/graphics/nvidia/graphics/nvidia-nvidia ; }
nvidia-cd(){  cd $(nvidia-dir); }
nvidia-mate(){ mate $(nvidia-dir) ; }
nvidia-get(){
   local dir=$(dirname $(nvidia-dir)) &&  mkdir -p $dir && cd $dir

}
