# === func-gen- : vr/vr fgp vr/vr.bash fgn vr fgh vr
vr-src(){      echo vr/vr.bash ; }
vr-source(){   echo ${BASH_SOURCE:-$(env-home)/$(vr-src)} ; }
vr-vi(){       vi $(vr-source) ; }
vr-env(){      elocal- ; }
vr-usage(){ cat << EOU

VR Interface Dev Experience
============================


* "Wild West of VR - Discovering the Rules of Oculus Rift Development"

* https://www.youtube.com/watch?v=_vqNpZqnl1o&index=27&list=PLckFgM6dUP2hc4iy-IdKFtqR9TeZWMPjm


VR SDKs
--------

* NVIDIA VRWorks
* http://www.pcgamer.com/amd-liquidvr-vs-nvidia-vrworks-the-sdk-wars/


Wireless Vive
---------------

* :google:`wireless vive`

* https://uploadvr.com/tpcast-wireless-vive-kit-works/
* https://techcrunch.com/2016/12/15/rivvr-brings-wireless-vr-to-the-oculus-rift-and-htc-vive/
* https://vrworld.com/2016/11/16/htc-2017-vive-design-wireless-vr/

* :google:`wireless vive tpcast rivvr`


NVIDIA Pascal VR Features
---------------------------

* https://developer.nvidia.com/pascal-vr-tech


NVIDIA VR Funhouse Source on Github
---------------------------------------


* https://developer.nvidia.com/vr-funhouse-source-and-editor-now-available-modders
* https://developer.nvidia.com/vr-funhouse-mod-kit
* https://developer.nvidia.com/gameworks-source-github

To gain access to source you need to agree to a license
and provide github username whilst logged in as NVIDIA registered
developer.

After 30 min got email with invite to https://github.com/NVIDIAGameWorks
providing a link to join.

::

    You’ve been added to the GameWorks_EULA_Access team for the NVIDIA GameWorks
    organization. GameWorks_EULA_Access has 5925 members and gives pull access to
    294 NVIDIAGameWorks repositories.

    View GameWorks_EULA_Access:
    https://github.com/orgs/NVIDIAGameWorks/teams/gameworks_eula_access

    Read more about team permissions here:
    https://help.github.com/articles/what-are-the-different-access-permissions






GTX 1080 specs
---------------

* http://www.geforce.com/hardware/10series/geforce-gtx-1080

Graphics Card Dimensions:

 4.376" Height
 10.5" Length
2-Slot Width

Thermal and Power Specs:
   94  Maximum GPU Temperature (in C)
 180W  Graphics Card Power (W)
 500W  Recommended System Power (W)3
8-Pin  Supplementary Power Connectors


VR System Requirements from NVIDIA 
-----------------------------------

* http://www.geforce.com/hardware/technology/vr/system-requirements

GPU: NVIDIA GeForce GTX 970 or greater
CPU: Intel Core i5-4590 equivalent or greater
Memory/RAM: 8GB+ RAM
Video Output: 1x HDMI 1.3
Ports: 3x USB 3.0
OS: Windows 7 SP1 (64bit) or higher
Driver: GeForce 359 or newer

HTC Vive Requirements
----------------------

* https://www.htcvive.com/us/product-optimized/

GPU: NVIDIA GeForce® GTX 970 or greater
CPU: Intel i5-4590 or greater   
RAM: 4GB+
Video Output: HDMI 1.4 or DisplayPort 1.2 or newer
USB Port: 1x USB 2.0 or greater port
Operating System: Windows 7 SP1 or newer



Oculus Recommended Laptops
---------------------------

* https://www.reddit.com/r/oculus/wiki/compatible_laptops

* :google:`Asus ROG GX700VO`
* :google:`Aorus X7 DT`
* :google:`MSI GT72S Dragon`


VR Capable Laptops
-------------------

* :google:`NVIDIA GTX 980 laptops`

* http://blogs.nvidia.com/blog/2015/09/22/notebooks/

  * all TW companies as is HTC

* :google:`Aorus ASUS MSI Clevo`
* :google:`Aorus X7 DT`  
* :google:`AORUS X7 G-SYNC`
* :google:`MSI GT72 G-SYNC`
* :google:`MSI GT80`
* :google:`ASUS GX700VO`
* :google:`Clevo P775DM`
* :google:`Clevo P870DM`

* http://www.anandtech.com/show/9649/nvidia-gtx-980-in-notebooks (Sept 2015)

  ..full desktop-class GTX 980 (with 2048 cores) will be made available in an MXM format for notebooks..

* http://arstechnica.com/gadgets/2015/09/nvidia-crams-desktop-gtx-980-gpu-into-monster-17-inch-notebooks/

  ...All the launch models also only come with 1080p displays...

  That's disappointing given the sheer graphics grunt of the GTX 980, which is
  more than capable of pushing 1440p or 4K visuals at high settings. That said,
  Asus has teased that some of its upcoming 17-inch gaming notebooks will feature
  a 4K option.


Acer
~~~~~

* http://www.laptopmag.com/articles/acer-predator-17x


Gigabyte/Aorus
~~~~~~~~~~~~~~~~

* http://www.digitimes.com/news/a20140604VL200.html


NewEgg gaming laptops by GPU
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* http://www.newegg.com/Gaming-Laptops/Category/ID-363




EOU
}
vr-dir(){ echo $(local-base)/env/vr/vr-vr ; }
vr-cd(){  cd $(vr-dir); }
vr-mate(){ mate $(vr-dir) ; }
vr-get(){
   local dir=$(dirname $(vr-dir)) &&  mkdir -p $dir && cd $dir

}
