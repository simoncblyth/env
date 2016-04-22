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



HTC Vive Requirements
----------------------

RECOMMENDED PC SPECS

GPU: NVIDIA GeForce® GTX 970 / AMD Radeon™ R9 290 equivalent or greater
CPU: Intel i5-4590 / AMD FX 8350 equivalent or greater   
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
