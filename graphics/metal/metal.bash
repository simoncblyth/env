# === func-gen- : graphics/metal/metal fgp graphics/metal/metal.bash fgn metal fgh graphics/metal
metal-src(){      echo graphics/metal/metal.bash ; }
metal-source(){   echo ${BASH_SOURCE:-$(env-home)/$(metal-src)} ; }
metal-vi(){       vi $(metal-source) ; }
metal-env(){      elocal- ; }
metal-usage(){ cat << EOU

Metal : close to the metal OpenGL ES alternative on iOS and OSX
=================================================================

Metal 2 (WWDC June 2017)
---------------------------

* https://developer.apple.com/metal/


mtlpp : Metal C++ Wrapper
---------------------------

* https://github.com/naleksiev/mtlpp

* https://forums.developer.apple.com/thread/18860


Geometry Shader equivalent ?
---------------------------------

* https://forums.developer.apple.com/thread/4818


External Graphics Development Kit
----------------------------------

* https://developer.apple.com/development-kit/external-graphics/

macOS High Sierra brings support for external graphics processors to the Mac
for the first time. The External Graphics Development Kit enables you to
develop and test demanding graphics-intensive apps, including VR content
creation, on any Mac with Thunderbolt 3 connectivity.

Apps that use Metal, OpenCL, and OpenGL can now take advantage of the increased
performance that external graphics processors can bring. The External Graphics
Development Kit includes everything you need to start optimizing advanced VR
and 3D apps on external graphics processors with macOS High Sierra.


* Sonnet external GPU chassis with Thunderbolt 3 and 350W power supply
* AMD Radeon RX 580 8GB graphics card
* Belkin USB-C to 4-port USB-A hub
* Promo code for $100 towards the purchase of HTC Vive VR headset**


* The External Graphics Development Kit requires a Mac with Thunderbolt 3 running 
  the latest beta version of macOS High Sierra.


Thunderbolt 3
~~~~~~~~~~~~~~~

* https://support.apple.com/en-us/HT207256

MacBook Pro models introduced in 2016 feature Thunderbolt 3 (USB-C) ports that
let you connect devices and displays, charge your computer, and provide power
to connected devices—all through one simple, compact USB-C connector.

MacBook Pro (15-inch, 2016), MacBook Pro (13-inch, 2016, Four Thunderbolt 3
Ports), and MacBook Pro (13-inch, 2016, Two Thunderbolt 3 Ports) come equipped
with Thunderbolt 3, an I/O technology that connects devices to your computer at
speeds up to 40 Gbps. Thunderbolt 3 combines data transfer, video output, and
charging capabilities in a single, compact connector.




Warren Moore
-------------

* http://metalbyexample.com/
* https://github.com/metal-by-example
* https://realm.io/news/3d-graphics-metal-swift/
* https://github.com/warrenm/slug-swift-metal/

Amund Tveit
-------------

* http://memkite.com/blog/2015/06/10/swift-and-metal-gpu-programming-on-osx-10-11-el-capitan/
* http://memkite.com/blog/2014/12/30/example-of-sharing-memory-between-gpu-and-cpu-with-swift-and-metal-for-ios8/
* https://github.com/atveit

Simon Gladman
--------------

* https://realm.io/news/swift-summit-simon-gladman-metal/
* https://github.com/FlexMonkey
* https://github.com/FlexMonkey/MetalKit-Particles
* https://github.com/FlexMonkey/ParticleLab





EOU
}
metal-dir(){ echo $(local-base)/env/graphics/metal/graphics/metal-metal ; }
metal-cd(){  cd $(metal-dir); }
metal-mate(){ mate $(metal-dir) ; }
metal-get(){
   local dir=$(dirname $(metal-dir)) &&  mkdir -p $dir && cd $dir

}
