Hardware
==============

Requirements
-------------

* host that supports CUDA 

  * http://docs.nvidia.com/cuda/index.html

Operating System
~~~~~~~~~~~~~~~~~~

* http://chroma.bitbucket.org/install/rhel.html  
* RHEL 6+ SL6+
* http://en.wikipedia.org/wiki/Red_Hat_Enterprise_Linux#RHEL_6

  * `6.4, also termed Update 4, 2013-02-21 (kernel 2.6.32-358)`

CUDA Drivers
~~~~~~~~~~~~

How to install Nvidia CUDA environment in RHEL 6

   * https://access.redhat.com/site/solutions/64300

Which Nvidia GPU ?
~~~~~~~~~~~~~~~~~~

I guess for trivial computational tasks like OP propagation (essentially 
intersecting lines with triangles) should look for the maximum number of cores 
at a reasonable cost.

* :google:`tesla quadro geforce`  

  * Just a marketing distinction perhaps, pick you price kinda thing ? Thermal characteristics also ?

      * tesla : servers
      * quadro : professional content creators
      * geforce : gamers 

  * http://www.heatonresearch.com/node/2487
  * http://www.videocardbenchmark.net/high_end_gpus.html
  * http://www.gpugrid.net/gpugrid_donations.php
     
      * asking for Nvidia GTX680 which is close to the early adopter price kink


PassMark - G3D Mark
High End Videocards - Updated 25th of September 2013

::

    GeForce GTX Titan    8,111       $999.99
    GeForce GTX 780      7,897       $649.99  (TDP 250W) hi TDP might present powering/cooling problem for chassis
    GeForce GTX 770      6,247       $399.99
    GeForce GTX 680      5,706       $333.08  (TDP 195W) launch price 500 USD <<<<< requested by  http://www.gpugrid.net/gpugrid_donations.php
    GeForce GTX 670      5,375       $270.07
    GeForce GTX 690      5,123       $999.99
    Radeon HD 7970       5,081       $299.99
    Radeon HD 7990       5,041       $619.99
    GeForce GTX 760      5,039       $242.98
    GeForce GTX 580      4,936       $339.99*
    GeForce GTX 660 Ti   4,693       $234.99
    Radeon HD 7950       4,628       $204.99
    FirePro W8000        4,533     $1,429.99
    GeForce GTX 570      4,390       $385.00


What GPUs are commonly used with Chroma ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

https://bitbucket.org/chroma/chroma/issue/2/pycuda_driverlogicerror-cumemhostalloc

::

   I typically am running Chroma on a GTX 580 or 680, but also GTX 470 and 550 Ti sometimes

   ..another machine with a Tesla C2075 with compute capability 2.0 


https://groups.google.com/forum/#!topic/chroma-sim/v1eePC45h5k

GPU Reviews
~~~~~~~~~~~~~

* https://developer.nvidia.com/cuda-gpus for CCCs
* http://www.anandtech.com/tag/gpus

* http://www.anandtech.com/show/7356/capsule-review-evga-geforce-gtx-780-superclocked-acx Sept 2013 review
* http://www.anandtech.com/show/6760/nvidias-geforce-gtx-titan-part-1  Feb 2013
* http://www.geforce.com/hardware/notebook-gpus/geforce-gt-750m

* http://en.wikipedia.org/wiki/GeForce_600_Series

  * http://www.anandtech.com/show/5699/nvidia-geforce-gtx-680-review March 2012 
  * http://www.geforce.com/hardware/desktop-gpus/geforce-gtx-680/specifications 
    1536 CUDA cores, 2048 MB, CCC 3.0

    98C       Maximum GPU Tempurature (in C)
    195W      Maximum Graphics Card Power (W)
    550W      Minimum System Power Requirement (W)5
    Two 6-pin Supplementary Power Connectors

* http://en.wikipedia.org/wiki/GeForce_700_Series

  * http://www.anandtech.com/show/6973/nvidia-geforce-gtx-780-review May 2013 [~650 USD]
  * http://www.nvidia.com/gtx-700-graphics-cards/gtx-780/  2304 CUDA cores, 3072 MB 

  * http://www.anandtech.com/show/6994/nvidia-geforce-gtx-770-review May 2013 [~400 USD]
  * http://www.nvidia.com/gtx-700-graphics-cards/gtx-770/  1536 CUDA cores, 2048 MB 

    GTX 770 is essentially GTX 680 on steroids. Higher core clockspeeds and memory clockspeeds give it performance exceeding GTX 680, 
    while higher voltages and a higher TDP allow it to clock higher and for it to matter. 

  * http://www.anandtech.com/show/7103/nvidia-geforce-gtx-760-review June 2013  [~250 USD]
  * http://www.nvidia.com/gtx-700-graphics-cards/gtx-760/  1152 CUDA cores, 2048 MB, 170W TDP



Costs
~~~~~~~

* http://www.newegg.com/Product/ProductList.aspx?Submit=ENE&DEPA=0&Order=BESTMATCH&Description=GeForce+GTX&N=100006662&isNodeId=1



Reference
----------

* :google:`database of NVIDIA GPU with CUDA compute capability`
* http://en.wikipedia.org/wiki/CUDA
* https://developer.nvidia.com/cuda-gpus

  * matrix of CCC for each GPU, but new releases are not there 

* http://www.geforce.com/hardware/technology/cuda/supported-gpus

NTU Planned GPU Cluster Upgrade
-------------------------------

::

    NVidia Kepler K20 GPU nodes and Intel Xeon Phi nodes



Apple rMBP and iMac with Mobile variants
------------------------------------------
::

    Mid 2012 rMBP 15 inch

    Nvidia GeForce GT 650M, CCC 3.0

    Sept 2013 iMac

    21.5 inch, $1499, 2.9 GHz quad-core Intel i5 and NVIDIA GeForce GT 750M with 1 GB video memory 
    [GeForce GT 750M  CCC 3.0, from https://developer.nvidia.com/cuda-gpus ]

    27 inch, $1799, 3.2 GHz quad-core Intel Core i5 and NVIDIA GeForce GT 755M with 1 GB video memory 
    [GeForce GT 755M   CCC ?  ]

    27 inch, $1999, 3.4 GHz quad-core Intel Core i5 and NVIDIA GeForce GTX 775M with 2 GB video memory 
    [GeForce GTX 775M CCC ? http://www.techpowerup.com/gpudb/2381/geforce-gtx-775m.html   rumored CCC 3.0]

    Configurable to NVIDIA GeForce GTX 780M with 4GB of GDDR5 memory.





