# === func-gen- : graphics/vr/vr fgp graphics/vr/vr.bash fgn vr fgh graphics/vr src base/func.bash
vr-source(){   echo ${BASH_SOURCE} ; }
vr-edir(){ echo $(dirname $(vr-source)) ; }
vr-ecd(){  cd $(vr-edir); }
vr-dir(){  echo $LOCAL_BASE/env/graphics/vr/vr ; }
vr-cd(){   cd $(vr-dir); }
vr-vi(){   vi $(vr-source) ; }
vr-env(){  elocal- ; }
vr-usage(){ cat << EOU

VR
===


Surveying available devices, Q1 2019
---------------------------------------

* https://venturebeat.com/2019/03/20/the-vr-headset-market-is-about-to-get-way-too-crowded-and-confusing/


Vive Cosmos (Q3 2019)
-----------------------

* https://skarredghost.com/2019/05/20/vive-cosmos-news-release-date/


Mobile Headsets : Oculus Quest vs Vive Focus Plus
---------------------------------------------------

* https://skarredghost.com/2019/05/27/oculus-quest-vs-htc-vive-focus-plus/

SDK
~~~~~~

* Oculus Utilities 
* Vive Wave SDK (SteamVR of mobile : works with several headsets) 

Ecosystem
~~~~~~~~~~

* Oculus : closed system
* Viveport : much more open, Chinese market

Passthrough
~~~~~~~~~~~~~~~

The fact that there are two front cameras in the same position of the eyes lets
the Focus have a working passthrough. The passthrough of the Quest is heavily
distorted (even if I really praise Oculus for what has been able to accomplish…
with the cameras in that position, it wasn’t easy to obtain a passthrough!),
while the one of the Focus is great and undistorted. On the Focus, you can also
access it through the SDK, while with Oculus this is impossible.

Summary
~~~~~~~~~~~~

So, the Focus is more open and offers more features and more hackability. The
Quest is closed and offers maybe a bit fewer features… but these ones are all
super-polished and work incredibly well.

Price
~~~~~~~~

The final price for the Oculus Quest is $399 (€449) for the 64 GB version and
$499 (€549) for the 128 GB one. The Vive Focus Plus costs $799. As you can see,
there is no competition: for the price of a Focus Plus, you can buy two Quests.

If we sum anyway to the headset the price of content, things may change a bit.
The Quest is a console, and as all consoles, it subsidize the hardware with the
sales of the software. The high-quality games on the Quest are all pricey and
most of them cost more than $20, with the top ones being $30 each.




EOU
}
vr-get(){
   local dir=$(dirname $(vr-dir)) &&  mkdir -p $dir && cd $dir

}
