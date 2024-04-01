# === func-gen- : graphics/nvidia/nvidia fgp graphics/nvidia/nvidia.bash fgn nvidia fgh graphics/nvidia
nvidia-src(){      echo graphics/nvidia/nvidia.bash ; }
nvidia-source(){   echo ${BASH_SOURCE:-$(env-home)/$(nvidia-src)} ; }
nvidia-vi(){       vi $(nvidia-source) ; }
nvidia-env(){      elocal- ; }
nvidia-usage(){ cat << EOU


NVIDIA
========


GPUs March 2024
----------------

* https://www.techpowerup.com/gpu-specs/rtx-5000-ada-generation.c4152
* https://resources.nvidia.com/en-us-design-viz-stories-ep/nvidia-global-workshteet?lx=CCKW39&contentType=news

3rd Generation RTX (Ada Lovelace) workstation GPUs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://www.pugetsystems.com/labs/articles/nvidia-rtx-ada-generation-content-creation-review/
* https://wccftech.com/us-adds-more-nvidia-gpus-to-china-ban-list-rtx-6000-ada-rtx-a6000-l4/

Comparing "NVIDIA RTX 5000 Ada Generation" with "NVIDIA RTX A5000 (Ampere Generation)"
shows 2x performance improvement in many metrics according to the below:

* https://www.leadtek.com/eng/news/product_news_detail/1717



3rd Generation RTX laptop GPUs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://www.anandtech.com/show/18781/nvidia-unveils-rtx-ada-lovelace-gpus-for-laptops-desktop-rtx-4000-sff


US Export to China Restrictions
---------------------------------

* https://www.semianalysis.com/p/wafer-wars-deciphering-latest-restrictions


Micro-benchmarking
--------------------

2018 : Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking

* https://arxiv.org/pdf/1804.06826.pdf

2019 : Low Overhead Instruction Latency Characterization for NVIDIA GPGPUs

* https://arxiv.org/pdf/1905.08778.pdf


Laptop GPUs check Oct 2023
----------------------------

* https://www.anandtech.com/show/18781/nvidia-unveils-rtx-ada-lovelace-gpus-for-laptops-desktop-rtx-4000-sff

* RTX 5000 Ada Laptop GPU : 9728 CUDA Cores : 16 GB VRAM : 03/2023

* :google:`RTX 5000 Ada Laptop GPU`

* https://www.notebookcheck.net/NVIDIA-RTX-5000-Ada-Generation-Laptop-GPU-GPU-Benchmarks-and-Specs.742159.0.html
* https://www.notebookcheck.net/HP-ZBook-Fury-16-G10-mobile-workstation-review-100-W-Nvidia-RTX-5000-Ada.751426.0.html
* https://www.notebookcheck.net/Dell-Precision-5680-review-Ada-Lovelace-dominates-on-workstations.741263.0.html


* https://www.notebookcheck.net/Nvidia-RTX-Studio-could-spell-trouble-for-the-long-running-HP-ZBook-Dell-Precision-and-Lenovo-ThinkPad-P-series.422999.0.html


06/04/2019
Traditionally, users who wanted a mobile workstation with Quadro or FirePro
graphics would almost always turn towards the HP ZBook, Lenovo ThinkPad P
series, or the Dell Precision series. These three OEMs control nearly the
entire mobile workstation market. While there are a very small handful of
exceptions like the Fujitsu Celsius, Clevo series, or MSI W series, none offer
anything close to the scale or warranties of the aforementioned big three. 

This is all set to change post Computex 2019 with the Nvidia Studio program.
Instead of limiting the mobile Quadro series to just three or four big OEMs,
the chipmaker will open up the platform to well-known makers of sleek and sexy
gaming laptops. Consumer-grade models like the Asus ZenBook Pro, Gigabyte Aero
15, Alienware m15, and even the Razer Blade series will begin offering Quadro
GPUs to complement the standard GeForce GPU SKUs. Workstation users will now
have a much larger pool of lighter, more portable, and arguably more visually
attractive options to choose from that is only expected to grow bigger over
time. 


* https://www.notebookcheck.net/MSI-Stealth-17-Studio-review-A-laptop-with-a-quiet-RTX-4080-for-almost-every-occasion.708750.0.html





Cluster Management
---------------------

* https://github.com/NVIDIA/deepops
* https://www.nvidia.com/en-us/data-center/bright-cluster-manager/
* https://www.run.ai/guides/slurm/understanding-slurm-gpu-management



nvml : NVIDIA Management Library (C API)
------------------------------------------

* see nvml-

* https://developer.nvidia.com/nvidia-management-library-nvml




nvidia-ml-py : Python Bindings for the NVIDIA Management Library
------------------------------------------------------------------

* https://pypi.org/project/nvidia-ml-py/#history

nvidia-smi
------------

* https://stackoverflow.com/questions/8223811/a-top-like-utility-for-monitoring-cuda-activity-on-a-gpu

::

    nvidia-smi -l 1        # every second
    nvidia-smi -lms 500    # every half second  
    watch -n0.2 nvidia-smi # every 0.2 second, but forking process every time : probably unwise


nvidia-visual-profiler 
----------------------

https://developer.nvidia.com/nvidia-visual-profiler

https://developer.nvidia.com/nvidia-cuda-toolkit-11_7_0-developer-tools-mac-hosts

https://developer.nvidia.com/nsight-compute



SER : Shader Execution Reordering : via NVAPI
-------------------------------------------------

* https://developer.nvidia.com/rtx/path-tracing/nvapi/get-started
* https://developer.nvidia.com/sites/default/files/akamai/gameworks/ser-whitepaper.pdf
* ~/opticks_refs/nvidia-ser-whitepaper.pdf

Shader Execution Reordering (SER) is a new scheduling technology introduced
with the Ada Lovelace generation of NVIDIA GPUs. It is highly effective at
simultaneously reducing both execution divergence and data divergence. SER
achieves this by on-the-fly reordering threads across the GPU such that groups
of threads perform similar work and therefore use GPU resources more
efficiently. This happens with minimal overhead: the Ada hardware architecture
was designed with SER in mind and includes optimizations to the SM and memory
system specifically targeted at efficient thread reordering. Using SER, we
observe speedups of up to 2x in raytracing regimes of real-world applications,
achieved with only a small amount of developer effort. To applications, SER is
exposed through a small API that gives developers new flexibility and full
control over where in their shaders reordering will happen. This API is
detailed in the following sections.

The API concepts described above will be available for Microsoft DirectX 12
(via NVAPI), Vulkan (via vendor extension), and OptiX. The following reference
covers NVAPI. For Vulkan and OptiX, please refer to the respective API
documentation once it becomes available.

The SER NVAPI is supported on all raytracing-capable NVIDIA GPUs starting with
R520 drivers. Vulkan and OptiX support will be added with later releases.


subwarp interleaving
---------------------

* https://research.nvidia.com/publication/2022-01_GPU-Subwarp-Interleaving
* ~/opticks_refs/Damani_Subwarp_Interleaving_HPCA_IT_2022_0.pdf


nvcc compile hang from __forceinline__
------------------------------------------

* https://forums.developer.nvidia.com/t/nvcc-hanging-compiling-a-single-cu-to-ptex/156838



A40 Cards
-----------

* https://www.anandtech.com/show/16137/nvidia-announces-ampere-rtx-a6000-a40-cards-for-pro-viz

  Quadro No More? NVIDIA Announces Ampere-based RTX A6000 & A40 Video Cards For Pro Visualization
  Oct 2020 

* https://www.nvidia.com/en-gb/data-center/a40/





vCHEP 2021 : C.Legett : Porting HEP Parameterized Calorimeter Simulation Code to GPUs
----------------------------------------------------------------------------------------

* ~/opticks_refs/FastCaloSim_for_vCHEP_2021_f.pdf
* https://indico.cern.ch/event/948465/contributions/4323701/

Can run multiple concurrent process all sharing one (or more) GPUs

* use **nvidia-cuda-mps-server** to share 2 P100s between up to 32 processes
* device based time slicing of GPU
* curve is mostly flat – nowhere near saturating GPU resources
* can run 62 processes on a V100 w/ 48GB with little impact on performance

nvidia-cuda-mps-server
-------------------------

* https://docs.nvidia.com/deploy/pdf/CUDA_Multi_Process_Service_Overview.pdf


hardware probing : lspci and dmidecode
---------------------------------------

::

   lspci | grep -i nv



AWS : G4 instance (with NVIDIA T4)
------------------------------------

* Amazon is launching a new instance, the G4 instance, which runs Tesla T4 accelerators


Rapids : collection of machine learning libs incubated by NVIDIA
-----------------------------------------------------------------

* https://www.developer.nvidia.com/rapids
* https://rapids.ai

::

   cuDF
   cuML
   cuGraph


vGPU : Quadro vDWS (Quadro Virtual Data Center Workstation) : required for CUDA
----------------------------------------------------------------------------------

See vgpu-



Issue : disabling G-SYNC gives black screen, reboot doesnt fix
----------------------------------------------------------------

nvidia-settings over ssh ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* disabling G-SYNC using display panel makes screen go black
* tried using "nvidia-settings" over ssh but seems not to be able to control target X server
  over ssh, although XQuartz started up on laptop::

    epsilon:~ blyth$ ssh J -X
    Last login: Fri Jul 20 13:19:00 2018 from 10.10.2.91
    [blyth@localhost ~]$ nvidia-settings -q AllowGSYNC

    ERROR: Unable to load info from any available system

* presumably need some config to talk to the X server on the machine


Looks credible : way to talk to Xserver over ssh 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

https://devtalk.nvidia.com/default/topic/1032741/linux/tuning-nvidia-settings-over-ssh-error/

Ok, you're using GDM/Gnome, so when you start your system GDM spawns its own
Xserver. You'll need the right XAUTHORITY set, as mentioned in the link you
gave but that was for lightdm.  Run


::

    [blyth@localhost ~]$ ps aux | grep X
    root       1256  0.0  0.0 221232  4716 ?        Ss   14:20   0:00 /usr/bin/abrt-watch-log -F Backtrace /var/log/Xorg.0.log -- /usr/bin/abrt-dump-xorg -xD
    root       1745 26.0  0.1 318612 72476 tty1     Rsl+ 14:20  12:14 /usr/bin/X :0 -background none -noreset -audit 4 -verbose -auth /run/gdm/auth-for-gdm-blah-blah/database -seat seat0 -nolisten tcp vt1


::

    [blyth@localhost ~]$ XAUTHORITY=/run/gdm/auth-for-gdm-blah-blah/database DISPLAY=:0 nvidia-settings -q all

    Attributes queryable via localhost.localdomain:0.0:

      Attribute 'OperatingSystem' (localhost.localdomain:0.0): 0.
        The valid values for 'OperatingSystem' are in the range 0 - 2 (inclusive).
        'OperatingSystem' is a read-only attribute.
        'OperatingSystem' can use the following target types: X Screen, GPU.

      Attribute 'NvidiaDriverVersion' (localhost.localdomain:0.0): 396.26 
        'NvidiaDriverVersion' is a string attribute.
        'NvidiaDriverVersion' is a read-only attribute.
        'NvidiaDriverVersion' can use the following target types: X Screen, GPU.

     ...

    [blyth@localhost ~]$ XAUTHORITY=/run/gdm/auth-for-gdm-0isL8H/database DISPLAY=:0 nvidia-settings -q ALLOWGSYNC

      Attribute 'AllowGSYNC' (localhost.localdomain:0.0): 0.
        'AllowGSYNC' is a boolean attribute; valid values are: 1 (on/true) and 0 (off/false).
        'AllowGSYNC' can use the following target types: X Screen.

    [blyth@localhost ~]$ XAUTHORITY=/run/gdm/auth-for-gdm-0isL8H/database DISPLAY=:0 nvidia-settings -q ShowGSYNCVisualIndicator

      Attribute 'ShowGSYNCVisualIndicator' (localhost.localdomain:0.0): 0.
        'ShowGSYNCVisualIndicator' is a boolean attribute; valid values are: 1 (on/true) and 0 (off/false).
        'ShowGSYNCVisualIndicator' can use the following target types: X Screen.







* https://thebravestatistician.wordpress.com/2017/08/13/tweaking-your-nvidia-gpu-via-ssh-using-nvidia-settings/




Libraries
------------

CUBLAS, CUSOLVER and MAGMA by example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://developer.nvidia.com/sites/default/files/akamai/cuda/files/Misc/mygpu.pdf

* http://docs.nvidia.com/cuda/pdf/CUSOLVER_Library.pdf



NVIDIA GPU CLOUD
-----------------

* https://docs.nvidia.com/ngc/ngc-introduction/index.html
* https://www.nvidia.com/en-us/gpu-cloud/


NVIDIA V100
-------------

* https://www.anandtech.com/show/11367/nvidia-volta-unveiled-gv100-gpu-and-tesla-v100-accelerator-announced


NVIDIA RTX at GTC 2018
------------------------

* https://developer.nvidia.com/rtx

NVIDIA RTX is the product of 10 years of work in computer graphics algorithms
and GPU architectures. It consists of a highly scalable ray tracing technology
running on NVIDIA Volta architecture GPUs. Developers can access NVIDIA RTX
technology through the NVIDIA OptiX application programming interface, through
Microsoft’s new DirectX Raytracing API and, soon, Vulkan, the new generation,
cross-platform graphics standard.

...

10 years ago OptiX introduced the programmable shader model for ray tracing
(OptiX GPU Ray Tracing ACM paper). NVIDIA has continued to invest in hardware,
software and algorithms to accelerate that programming model on our GPUs.

The OptiX API is an application framework that leverages RTX Technology to
achieve optimal ray tracing performance on the GPU. It provides a simple,
recursive, and flexible pipeline for accelerating ray tracing algorithms.
Additionally the post processing API includes an AI-accelerated denoiser, which
also leverages RTX Technology. The post processing API can be used
independently from the ray tracing portion of the pipeline.



NVIDIA pushing Volta, or is it really needed for RTX ?
--------------------------------------------------------

* https://devblogs.nvidia.com/introduction-nvidia-rtx-directx-raytracing/

So will ray tracing always remain a dream of the future, and never arrive in
the present? At GDC 2018, NVIDIA unveiled RTX, a high-performance
implementation that will power all ray tracing APIs supported by NVIDIA on
Volta and future GPUs. At the same event, Microsoft announced the integration
of ray tracing as a first-class citizen into their industry standard DirectX
API.

Putting these two technologies together forms such a powerful combination that
we can confidently answer the above question: the future is here! This is not a
hyperbole: leading game studios are developing upcoming titles using RTX
through DirectX — today. Ray tracing in games is no longer a pipe dream. It’s
happening, and it will usher in a new era of real-time graphics.

* https://developer.nvidia.com/gameworks-ray-tracing


NVIDIA RTX hardware implementation (from Volta) exposed by DXR
----------------------------------------------------------------

* see dxr-

* https://www.anandtech.com/show/12546/nvidia-unveils-rtx-technology-real-time-ray-tracing-acceleration-for-volta-gpus-and-later

In conjunction with Microsoft’s new DirectX Raytracing (DXR) API announcement,
today NVIDIA is unveiling their RTX technology, providing ray tracing
acceleration for Volta and later GPUs. Intended to enable real-time ray tracing
for games and other applications, RTX is essentially NVIDIA's DXR backend
implementation. For this NVIDIA is utilizing a mix of software and hardware –
including new microarchitectural features – though the company is not
disclosing further details. Alongside RTX, NVIDIA is also announcing their new
GameWorks ray tracing tools, currently in early access to select development
partners.

With NVIDIA working with Microsoft, RTX is fully supported by DXR, meaning that
all RTX functionality is exposed through the API. And while only Volta and
newer architectures have the specific hardware features required for hardware
acceleration of DXR/RTX, DXR's compatibility mode means that a DirectCompute
path will be available for non-Volta hardware. Beyond Microsoft, a number of
developers and game engines are supporting RTX, with DXR and RTX tech demos at
GDC 2018.

On that note, since the entire “GPU – RTX – DXR – GameWorks Ray Tracing” stack
only applies to Volta, the broader public is essentially limited to the Titan
V, and NVIDIA likewise noted that RTX technology of present was primarily
intended for developer use. For possible ray tracing acceleration on pre-Volta
architectures, NVIDIA only referred back to DXR, though Microsoft has equally
referred back to vendors for hardware-related technical details. And while
strict performance numbers aren’t being disclosed, NVIDIA stated that real time
ray tracing with RTX on Volta would be “integer multiples faster” than with DXR
on older hardware.






Shadowplay
-----------

Video capture from NVIDIA

* http://www.geforce.com/geforce-experience/shadowplay


Install Logging Linux
----------------------

* /var/log/nvidia-installer.log 

::

     .1 nvidia-installer log file '/var/log/nvidia-installer.log'
      2 creation time: Wed Mar  8 11:03:32 2017
      3 installer version: 367.48
      4 
      5 PATH: /usr/lib64/qt-3.3/bin:/usr/local/sbin:/usr/local/bin:/sbin:/bin:/usr/sbin:/usr/bin:/root/bin
      6 
      7 nvidia-installer command line:
      8     ./nvidia-installer
      9     --ui=none
     10     --no-questions
     11     --accept-license
     12     --disable-nouveau
     13     --run-nvidia-xconfig
     14 
     ...



Titan V : late 2017, 5120 CUDA cores, 3000 USD compute monster, still able to do graphics
-------------------------------------------------------------------------------------------

* https://www.anandtech.com/show/12170/nvidia-titan-v-preview-titanomachy


NPP : NVIDIA Performance Primitives
-------------------------------------

* https://developer.nvidia.com/npp

GPU-accelerated image, video, and signal processing functions that perform up to 30x faster than CPU-only implementations


Mac Driver, eGPU
--------------------

* http://www.macworld.com/article/3189405/macs/nvidia-releases-mac-driver-with-support-for-titan-xp-and-geforce-gtx-1000-series.html
* https://9to5mac.com/2017/04/11/hands-on-powering-the-macbook-pro-with-an-egpu-using-nvidias-new-pascal-drivers/

Gameworks
-----------

* http://docs.nvidia.com/gameworks/

eGPU
-----

* requires thunderbolt 3 

Thoughts
    eGPU boxes are essentially a PC chassis with big powersupply and GPU slots... it would
    be better to find way to use the GPU from an actual Window/Linux PC 
    (sorta like the old Firewire target disk mode) rather than purchasing a single trick device  


* :google:`Akitio Thunder3 eGPU`
* :google:`Akitio Node`

* https://gpunerd.com/external-gpu-buyers-guide-egpu
* https://9to5mac.com/2017/04/11/hands-on-powering-the-macbook-pro-with-an-egpu-using-nvidias-new-pascal-drivers/
* https://9to5mac.com/2017/04/19/akitio-node-gtx-1080-ti-gpu-macbook-pro-gaming-egpu/


NVIDIA Pascal Drivers for Mac OS
---------------------------------

* http://www.anandtech.com/show/11254/nvidia-to-release-pascal-drivers-for-macos


Apple External GPU (WWDC June 2017)
-------------------------------------

APPLE TO SUPPORT EXTERNAL GRAPHICS CARD ENCLOSURES ON MACBOOK PROS, IMACS

* https://www.digitaltrends.com/computing/apple-external-graphics-card-enclosure-support-soon/

* http://appleinsider.com/articles/17/06/06/apple-egpu-dev-kit-requires-external-monitor-nothing-unique-about-graphics-card-or-enclosure

Apple has started taking orders for its external GPU developer's kit for
examination with macOS High Sierra, but the Apple software available to the
wider user base until spring 2018, as released still has major caveats, and
mandates an external monitor or VR kit for use.


Apple Metal 2 External Graphics Development Kit  
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://egpu.io/forums/thunderbolt-enclosures/apple-metal-2-external-graphics-development-kit/

* Sonnet Breakaway Box enclosure
* AMD Radeon RX 580 GPU
* Belkin USB-C to 4-port USB-A hub


G-Sync
-------

* https://en.m.wikipedia.org/wiki/Nvidia_G-Sync

G-Sync enabled monitors allow screen updates
to be driven by the NVIDIA GPU, 
avoiding tearing or stuttering (from V-Sync).

It is particularly useful in keeping the interface 
usable when frame rates drop below target. 


Disabling GPU Boost for deterministic benchmarking
----------------------------------------------------

* https://developer.nvidia.com/setstablepowerstateexe-%20disabling%20-gpu-boost-windows-10-getting-more-deterministic-timestamp-queries


Capture SDK (formerly Grid SDK)
---------------------------------

Remote 

* https://developer.nvidia.com/capture-sdk
* https://aws.amazon.com/blogs/aws/new-g2-instance-type-with-4x-more-gpu-power/

10 Series
-----------

Geforce GTX 1070
~~~~~~~~~~~~~~~~~~~

* http://www.anandtech.com/show/10336/nvidia-posts-full-geforce-gtx-1070-specs


Geforce GTX 1080
~~~~~~~~~~~~~~~~~~~~~~~~~

* https://developer.nvidia.com/introducing-nvidia-geforce-gtx-1080
* http://www.geforce.com/hardware/10series/geforce-gtx-1080
* http://international.download.nvidia.com/geforce-com/international/pdfs/GeForce_GTX_1080_Whitepaper_FINAL.pdf
* https://www.pugetsystems.com/labs/hpc/GTX-1080-CUDA-performance-on-Linux-Ubuntu-16-04-preliminary-results-nbody-and-NAMD-803/

* http://arstechnica.co.uk/gadgets/2016/05/nvidia-gtx-1080-1070-pascal-specs-price-release-date/

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


Ancient GPU On M : GeForce 8400 GS (Version 341.95)
----------------------------------------------------

* https://en.wikipedia.org/wiki/List_of_Nvidia_graphics_processing_units#GeForce_8_.288xxx.29_Series
* https://en.wikipedia.org/wiki/GeForce_8_series

  Note discrepancies in above sources.

  June 15, 2007
  Suggests OpenGL 3.3 capable


Graphics on Tesla GPUs 
-------------------------

* http://devblogs.nvidia.com/parallelforall/interactive-supercomputing-in-situ-visualization-tesla-gpus/

GPU Achitecture history
------------------------

* Fermi
* Kepler
* Maxwell
* Pascal 
* Volta
* Turing 

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



Server Side Rendering
-----------------------


There are ways of using remote viz with GPU server machines
connected to clients, however my impression is that these are 
complicated to setup, even requiring some development work.
So it is much easier to develop/debug locally. 

* http://www.nvidia.com/content/PDF/remote-viz-tesla-gpus.pdf
* https://devblogs.nvidia.com/parallelforall/linking-opengl-server-side-rendering/
* https://devblogs.nvidia.com/parallelforall/hpc-visualization-nvidia-tesla-gpus/


* https://devblogs.nvidia.com/parallelforall/egl-eye-opengl-visualization-without-x-server/
* http://on-demand.gputechconf.com/gtc/2015/presentation/S5660-Peter-Messmer.pdf


EGL
~~~~

* https://devblogs.nvidia.com/parallelforall/egl-eye-opengl-visualization-without-x-server/

With the release of NVIDIA Driver 355, full (desktop) OpenGL is now available
on every GPU-enabled system, with or without a running X server. The latest
driver (358) enables multi-GPU rendering support.

GLX and EGL use the Same OpenGL
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://devblogs.nvidia.com/parallelforall/linking-opengl-server-side-rendering/

The separation of OpenGL functions and context management functions into
separate libraries allows developers to build applications supporting multiple
context creation mechanisms.

For instance, this enables you to add an EGL backend to your glX-based
application to deliver cloud-based rendering capabilities. You need to modify
the context creation mechanism, as described in a previous post, to initialize
either glX or EGL. All the other rendering code remains the same. But now you
need to link against libOpenGL.so, libGLX.so, and libEGL.so.

If your application uses OpenGL extension functions, it is your responsibility
to use the correct extension function loader mechanism for the initialized
context. A GLX-based context should use glXGetProcAddress, whereas an EGL-based
context should use eglGetProcAddress. Note that the use of a particular loader
may be implicit: GLEW, for example, uses the GLEW_EGL C preprocessor macro to
choose the loader.


GLVND support in CMake
~~~~~~~~~~~~~~~~~~~~~~~~




For larger projects you probably don’t want to add the different libraries
manually, but rather delegate this to a build system like Kitware’s CMake.
Starting with version 3.10.0, CMake supports GLVND natively through its
FindOpenGL module. To utilize a specific context library, just specify it in
COMPONENTS and use the appropriate import targets. For example, the following
snippet compiles an application with support for both EGL and GLX contexts.

::

    find_package(OpenGL REQUIRED COMPONENTS OpenGL EGL GLX)
    add_executable(your_binary_name main.c)
    target_link_libraries(your_binary_name PRIVATE OpenGL::OpenGL OpenGL::EGL OpenGL::GLX)


To link against EGL only, simply drop GLX from COMPONENTS and remove OpenGL::GLX as an import target in target_link_libraries.

::

    find_package(OpenGL REQUIRED COMPONENTS OpenGL EGL)
    add_executable(your_binary_name main.c)
    target_link_libraries(your_binary_name PRIVATE OpenGL::OpenGL OpenGL::EGL)


GLVND : The GL Vendor-Neutral Dispatch library
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://github.com/NVIDIA/libglvnd
* https://github.com/NVIDIA/libglvnd/issues/63


OpenGL without X using EGL
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://devtalk.nvidia.com/default/topic/1005748/opengl/opengl-without-x-using-egl/

* https://github.com/VirtualGL/virtualgl/issues/10

* https://gist.github.com/dcommander/ee1247362201552b2532

Movie from images
~~~~~~~~~~~~~~~~~~~~

* https://superuser.com/questions/249101/how-can-i-combine-30-000-images-into-a-timelapse-movie
* http://fixounet.free.fr/avidemux/
* http://mariovalle.name/mencoder/mencoder.html

Learn more about server-side rendering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://devblogs.nvidia.com/parallelforall/linking-opengl-server-side-rendering/

The preferred way to use OpenGL on a headless server is via EGL, obviating the
need for an X server. In addition to modifying your application’s context
creation mechanism, this requires using the new GLVND ABI. Using GLVND is as
simple as changing your application to link against libOpenGL and libEGL
instead of libGL.

GLVND was first enabled in NVIDIA driver version 361.28, released in February
2016 and now widely available. In addition to an EGL ABI, the library also
allows for a number of OpenGL implementations to live side-by-side on a system,
simplifying deployment for your OpenGL applications.

Server-side rendering offers a range of advantages for large-scale
visualization, including reduced data transfer cost, simplified application
deployment and support for a wider client base.




::

    [simon@GPU cuda-8.0-samples]$ nvidia-smi
    Thu Sep  7 19:31:34 2017       
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 367.48                 Driver Version: 367.48                    |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |===============================+======================+======================|
    |   0  Tesla K80           Off  | 0000:05:00.0     Off |                    0 |
    | N/A   39C    P0    56W / 149W |      0MiB / 11439MiB |      0%      Default |
    +-------------------------------+----------------------+----------------------+
    |   1  Tesla K80           Off  | 0000:06:00.0     Off |                    0 |
    | N/A   32C    P0    66W / 149W |      0MiB / 11439MiB |      0%      Default |
    +-------------------------------+----------------------+----------------------+
    |   2  Tesla K80           Off  | 0000:84:00.0     Off |                    0 |
    | N/A   32C    P0    56W / 149W |      0MiB / 11439MiB |      0%      Default |
    +-------------------------------+----------------------+----------------------+
    |   3  Tesla K80           Off  | 0000:85:00.0     Off |                    0 |
    | N/A   31C    P0    73W / 149W |      0MiB / 11439MiB |     99%      Default |
    +-------------------------------+----------------------+----------------------+



Versions
-----------

    
============  ==========================  ======= =====================  ========
Tag            GPU                         Driver  CUDA Driver/Runtime    OptiX
============  ==========================  ======= =====================  ========
Tao            Quadro Maxell 2000 Mobile   375.39   8.0/7.5               4.0.0
junogpu002     Tesla 4*K40m                384.66   9.0/7.5               4.1.1
SDU server     Tesla 4*K80                 367.48   8.0/8.0             
Axel           NVIDIA Quadro M5000
============  ==========================  ======= =====================  ========



nvenc : hardware video encode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://developer.nvidia.com/nvidia-video-codec-sdk

NVIDIA Linux display driver 378.13 or newer
NVIDIA Windows display driver 378.66 or newer 




SDU 4*K80
------------

* https://devtalk.nvidia.com/default/topic/525927/display-driver-failed-installation-with-cuda-5-0/

If use see:

GPU Operation Mode
        Current                 : N/A
        Pending                 : N/A

Then you're board doesn't support changing GOM and it's a compute only board


If you see:

GPU Operation Mode
        Current                 : Compute
        Pending                 : Compute

Than you can change GOM. Just switch to Administrator user 
(or just run console with Administrator privileges):

nvidia-smi --gom=ALL_ON



[simon@GPU cuda-8.0-samples]$ nvidia-smi --format=csv --query-gpu=gom.current
gom.current
[Not Supported]
[Not Supported]
[Not Supported]
[Not Supported]



::

    [simon@GPU cuda-8.0-samples]$ nvidia-smi -q 

    ==============NVSMI LOG==============

    Timestamp                           : Thu Sep  7 19:19:58 2017
    Driver Version                      : 367.48

    Attached GPUs                       : 4
    GPU 0000:05:00.0
        Product Name                    : Tesla K80
        Product Brand                   : Tesla
        Display Mode                    : Disabled
        Display Active                  : Disabled
        Persistence Mode                : Disabled
        Accounting Mode                 : Disabled
        Accounting Mode Buffer Size     : 1920
        Driver Model
            Current                     : N/A
            Pending                     : N/A
        Serial Number                   : 0323416134647
        GPU UUID                        : GPU-5d468d1c-4386-a379-6575-cfc0f222abfa
        Minor Number                    : 0
        VBIOS Version                   : 80.21.1F.00.01
        MultiGPU Board                  : Yes
        Board ID                        : 0x300
        GPU Part Number                 : 900-22080-0100-000
        Inforom Version
            Image Version               : 2080.0200.00.04




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


    [root@GPU ~]# lspci | grep -i nvidia
    05:00.0 3D controller: NVIDIA Corporation Device 102d (rev a1)
    06:00.0 3D controller: NVIDIA Corporation Device 102d (rev a1)
    84:00.0 3D controller: NVIDIA Corporation Device 102d (rev a1)
    85:00.0 3D controller: NVIDIA Corporation Device 102d (rev a1)
    [root@GPU ~]# 


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



SDU
-----

::

    

    RHEL 6.x    2.6.32  4.4.7   2.12


* https://devtalk.nvidia.com/default/topic/493290/multiple-cuda-versions-can-they-coexist-/

Multiple CUDA versions

* http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#axzz4ryi9lITf

::

    [root@GPU ~]# cat /etc/redhat-release 
    Scientific Linux release 6.5 (Carbon)

    [root@GPU ~]# uname -a
    Linux GPU 2.6.32-431.el6.x86_64 #1 SMP Thu Nov 21 13:35:52 CST 2013 x86_64 x86_64 x86_64 GNU/Linux

    [root@GPU ~]# gcc --version
    gcc (GCC) 4.4.7 20120313 (Red Hat 4.4.7-17)
    Copyright (C) 2010 Free Software Foundation, Inc.
    This is free software; see the source for copying conditions.  There is NO
    warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.


    [root@GPU ~]# cat /proc/version
    Linux version 2.6.32-431.el6.x86_64 (mockbuild@sl6.fnal.gov) (gcc version 4.4.7 20120313 (Red Hat 4.4.7-3) (GCC) ) #1 SMP Thu Nov 21 13:35:52 CST 2013

    [root@GPU ~]# uname -a
    Linux GPU 2.6.32-431.el6.x86_64 #1 SMP Thu Nov 21 13:35:52 CST 2013 x86_64 x86_64 x86_64 GNU/Linux

    [root@GPU ~]# cat /etc/redhat-release 
    Scientific Linux release 6.5 (Carbon)

    [root@GPU ~]# uname -r
    2.6.32-431.el6.x86_64


::

    [root@GPU ~]# nvidia-smi
    Thu Sep  7 16:58:37 2017       
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 367.48                 Driver Version: 367.48                    |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |===============================+======================+======================|
    |   0  Tesla K80           Off  | 0000:05:00.0     Off |                    0 |
    | N/A   39C    P0    57W / 149W |      0MiB / 11439MiB |      0%      Default |
    +-------------------------------+----------------------+----------------------+
    |   1  Tesla K80           Off  | 0000:06:00.0     Off |                    0 |
    | N/A   32C    P0    66W / 149W |      0MiB / 11439MiB |      0%      Default |
    +-------------------------------+----------------------+----------------------+
    |   2  Tesla K80           Off  | 0000:84:00.0     Off |                    0 |
    | N/A   32C    P0    57W / 149W |      0MiB / 11439MiB |      0%      Default |
    +-------------------------------+----------------------+----------------------+
    |   3  Tesla K80           Off  | 0000:85:00.0     Off |                    0 |
    | N/A   31C    P0    73W / 149W |      0MiB / 11439MiB |     99%      Default |
    +-------------------------------+----------------------+----------------------+
                                                                                   
    +-----------------------------------------------------------------------------+
    | Processes:                                                       GPU Memory |
    |  GPU       PID  Type  Process name                               Usage      |
    |=============================================================================|
    |  No running processes found                                                 |
    +-----------------------------------------------------------------------------+





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

nvidia-smi-loop(){ nvidia-smi --loop=1 ; }


