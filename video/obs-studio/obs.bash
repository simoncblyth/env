# === func-gen- : video/obs-studio/obs fgp video/obs-studio/obs.bash fgn obs fgh video/obs-studio
obs-src(){      echo video/obs-studio/obs.bash ; }
obs-source(){   echo ${BASH_SOURCE:-$(env-home)/$(obs-src)} ; }
obs-vi(){       vi $(obs-source) ; }
obs-env(){      elocal- ; }
obs-usage(){ cat << EOU

OBS : Open Broadcaster Software
=================================

* https://obsproject.com

* paid alternatives : Camtasia, Screenflow

Usage for screen recording
--------------------------

* https://photography.tutsplus.com/tutorials/obs-for-screen-recording-quick-start--cms-28549

Before you can start recording you'll also need to add a source inside this
scene. With the default Scene selected (it will be highlighted) click the +
button at the bottom of the panel labeled *Sources*, then select Screen Capture
on Linux or Display Capture on Mac and Windows.


macOS
-------

* https://github.com/obsproject/obs-studio/releases/download/21.1.1/obs-mac-21.1.1-installer.pkg
* GUI installer, states will take 354.7 MB, lands into /Applications/OBS.app

Wizard
~~~~~~~

* pick optimize for recording (not streaming)
* Base (canvas) resolution options::

  Use Current (1440x900)  [default]
  Display 1 (1440x900)
  1920x1080    << picked this << 
  1280x720

* FPS : Either 60 or 30, prefer 60 when possible


The wizard ran some tests, and concluded::

   Recording Encoder          : software (x264)
   Recording Quality          : High Quality, Medium File size
   Base (Canvas) Resolution   : 1920x1080 
   Output (Scaled) Resolution : 1280x720 
                         FPS  :  60 

In "Output" can pick Recording format::

   flv  (default)
   mp4
   mov  (picked)
   mkv
   ts 
   m3u8

   



::

    In [5]: 1440./900.
    Out[5]: 1.6

    In [1]: 1280./720.
    Out[1]: 1.7777777777777777

    In [4]: 1920./1080.
    Out[4]: 1.7777777777777777

    In [3]: 16./9.
    Out[3]: 1.7777777777777777

     




Linux
-------

* https://github.com/obsproject/obs-studio/wiki/Install-Instructions#linux



EOU
}
obs-dir(){ echo $(local-base)/env/video/obs-studio/video/obs-studio-obs ; }
obs-cd(){  cd $(obs-dir); }
obs-mate(){ mate $(obs-dir) ; }
obs-get(){
   local dir=$(dirname $(obs-dir)) &&  mkdir -p $dir && cd $dir

}
