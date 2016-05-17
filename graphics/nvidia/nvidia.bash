# === func-gen- : graphics/nvidia/nvidia fgp graphics/nvidia/nvidia.bash fgn nvidia fgh graphics/nvidia
nvidia-src(){      echo graphics/nvidia/nvidia.bash ; }
nvidia-source(){   echo ${BASH_SOURCE:-$(env-home)/$(nvidia-src)} ; }
nvidia-vi(){       vi $(nvidia-source) ; }
nvidia-env(){      elocal- ; }
nvidia-usage(){ cat << EOU


NVIDIA
========


10 Series
-----------

Geforce GTX 1080, 1070
~~~~~~~~~~~~~~~~~~~~~~~~~

* http://www.geforce.com/hardware/10series/geforce-gtx-1080

* 2560 CUDA cores
* 8 GB GDDR5X
* OpenGL 4.5
* Windows 7-10, Linux, FreeBSDx86
* Recommended System Power : 500 W



* GTX 1080 599 USD  May 27   
* GTX 1070 380 USD  June 10


Computex  May 31 - June 4
----------------------------

* https://www.computextaipei.com.tw/en_US/member/visitor/preregister.html


Graphics on Tesla GPUs 
-------------------------

* http://devblogs.nvidia.com/parallelforall/interactive-supercomputing-in-situ-visualization-tesla-gpus/

GPU Achitecture history
------------------------

* Fermi
* Kepler
* Maxwell
* Pascal 

* http://en.m.wikipedia.org/wiki/Kepler_(microarchitecture)
* http://en.m.wikipedia.org/wiki/Maxwell_(microarchitecture)


GeForce GT 750M (GK107 : Kepler Architecture)
-----------------------------------------------

* http://www.geforce.com/hardware/notebook-gpus/geforce-gt-750m/description

The GeForce GT 750M is a graphics card by NVIDIA, launched in January 2013.
Built on the 28 nm process, and based on the GK107 graphics processor.
It features 384 shading units, 32 texture mapping units and 16 ROPs. NVIDIA has
placed 2,048 MB GDDR5 memory on the card, which are connected using a 128-bit
memory interface. The GPU is operating at a frequency of 941 MHz, which can be
boosted up to 967 MHz, memory is running at 1000 MHz. 


Tesla K20  (GK110 : Kepler Architecture)
------------------------------------------

* http://www.anandtech.com/show/6446/nvidia-launches-tesla-k20-k20x-gk110-arrives-at-last

K20c
~~~~~~

* http://www8.hp.com/h20195/v2/getpdf.aspx/c04111061.pdf?ver=2
* compute only, not capable of OpenGL 





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


Checking NVIDIA Driver and CUDA Versions On Linux
-----------------------------------------------------

* http://stackoverflow.com/questions/13125714/how-to-get-the-nvidia-driver-version-from-the-command-line

::

    -bash-4.1$ nvidia-smi
    Wed Feb  4 15:11:29 2015       
    +------------------------------------------------------+                       
    | NVIDIA-SMI 5.319.37   Driver Version: 319.37         |                       
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |===============================+======================+======================|
    |   0  Tesla K20m          Off  | 0000:03:00.0     Off |                    0 |
    | N/A   23C    P0    34W / 225W |       11MB /  4799MB |      0%      Default |
    +-------------------------------+----------------------+----------------------+
    |   1  Tesla K20m          Off  | 0000:84:00.0     Off |                    0 |
    | N/A   22C    P0    41W / 225W |       11MB /  4799MB |     77%      Default |
    +-------------------------------+----------------------+----------------------+
                                                                                   
    +-----------------------------------------------------------------------------+
    | Compute processes:                                               GPU Memory |
    |  GPU       PID  Process name                                     Usage      |
    |=============================================================================|
    |  No running compute processes found                                         |
    +-----------------------------------------------------------------------------+


    ## version of the currently loaded NVIDIA kernel module

    -bash-4.1$ cat /proc/driver/nvidia/version
    NVRM version: NVIDIA UNIX x86_64 Kernel Module  319.37  Wed Jul  3 17:08:50 PDT 2013
    GCC version:  gcc version 4.4.7 20120313 (Red Hat 4.4.7-4) (GCC) 

    -bash-4.1$ cuda-
    -bash-4.1$ nvcc --version
    nvcc: NVIDIA (R) Cuda compiler driver
    Copyright (c) 2005-2013 NVIDIA Corporation
    Built on Wed_Jul_17_18:36:13_PDT_2013
    Cuda compilation tools, release 5.5, V5.5.0
    -bash-4.1$ 



EOU
}
nvidia-dir(){ echo $(local-base)/env/graphics/nvidia/graphics/nvidia-nvidia ; }
nvidia-cd(){  cd $(nvidia-dir); }
nvidia-mate(){ mate $(nvidia-dir) ; }
nvidia-get(){
   local dir=$(dirname $(nvidia-dir)) &&  mkdir -p $dir && cd $dir

}

nvidia-gom(){
   nvidia-smi --format=csv --query-gpu=gom.current
}


nvidia-version(){
   type $FUNCNAME
   cat /proc/driver/nvidia/version      # Linux Only 
}


