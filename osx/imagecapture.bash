# === func-gen- : osx/imagecapture fgp osx/imagecapture.bash fgn imagecapture fgh osx
imagecapture-src(){      echo osx/imagecapture.bash ; }
imagecapture-source(){   echo ${BASH_SOURCE:-$(env-home)/$(imagecapture-src)} ; }
imagecapture-vi(){       vi $(imagecapture-source) ; }
imagecapture-env(){      elocal- ; }
imagecapture-usage(){ cat << EOU

ImageCapture
=============

* :google:`OSX prevent Camera Upload on connecting iPod`

* https://support.apple.com/en-is/HT201399
* http://www.tekrevue.com/tip/stop-iphoto-from-auto-launching-when-you-connect-your-iphone/


Prevent iPhoto from opening when connecting iPod to MBP
using ImageCapture.app ?

#. Disconnect and then connect the iPod, it should show up
   in sidebar of ImageCapture with the images.  

#. Select the device.

#. Click dislosure triangle at extreme bottom left to configure
   what happens on connecting the selected device.
   Choice of: 
 
   * No application
   * iPhoto.app
   * Image Capture.app
   * Preview.app
   * AutoImporter.app
   * Other...


Above method only worked after selecting "Always iPhoto" 
and then deselecting in "iPhoto"



EOU
}
imagecapture-dir(){ echo $(local-base)/env/osx/osx-imagecapture ; }
imagecapture-cd(){  cd $(imagecapture-dir); }
imagecapture-mate(){ mate $(imagecapture-dir) ; }
imagecapture-get(){
   local dir=$(dirname $(imagecapture-dir)) &&  mkdir -p $dir && cd $dir

}
