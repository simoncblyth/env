# === func-gen- : osx/macos fgp osx/macos.bash fgn macos fgh osx
macos-src(){      echo osx/macos.bash ; }
macos-source(){   echo ${BASH_SOURCE:-$(env-home)/$(macos-src)} ; }
macos-vi(){       vi $(macos-source) ; }
macos-env(){      elocal- ; }
macos-usage(){ cat << EOU

macOS
========

History
---------

* https://en.wikipedia.org/wiki/MacOS_version_history

==================== ======  ====================
macOS High Sierra     10.13   September 25, 2017
macOS Sierra          10.12   September 20, 2016
OS X El Capitan       10.11   September 30, 2015
OS X Yosemite         10.10   October 16, 2014
OS X Mavericks        10.9    October 22, 2013
==================== ======  ====================


macOS High Sierra 10.13.2 December 6 2017 Size 4.8 GB
---------------------------------------------------------

* https://itunes.apple.com/tw/app/macos-high-sierra/id1246284741?l=en&mt=12
* Metal 2

    

Mac App Store Install
-----------------------

Very quick download, must be a shim::

    simon:~ blyth$ du -hs /Applications/Install\ macOS\ High\ Sierra.app
     13M    /Applications/Install macOS High Sierra.app

    delta:~ blyth$ l /Applications/Install\ macOS\ High\ Sierra.app/Contents/Resources/
    total 14800
    -rw-r--r--  1 root  wheel      821 Jan  8 14:14 Info-Internal.plist
    -rwxr-xr-x  1 root  wheel   173856 Jan  8 14:14 InstallAssistantTool
    -rw-r--r--  1 root  wheel    11048 Jan  8 14:14 arrowbutton.tiff
    -rw-r--r--  1 root  wheel    11266 Jan  8 14:14 arrowbuttonFocus.tiff
    ...

     
::

    delta:~ blyth$ /Applications/Install\ macOS\ High\ Sierra.app/Contents/Resources/createinstallmedia -h
    Usage: createinstallmedia --volume <path to volume to convert>

    Arguments
    --volume, A path to a volume that can be unmounted and erased to create the install media.
    --applicationpath, A path to copy of the OS installer application to create the bootable media from.
    --nointeraction, Erase the disk pointed to by volume without prompting for confirmation.

    Example: createinstallmedia --volume /Volumes/Untitled

    This tool must be run as root.
    delta:~ blyth$ 
       


Sonnet eGPU breakaway box instructions for Sierra, not yet High Sierra
--------------------------------------------------------------------------

* http://www.sonnettech.com/support/downloads/manuals/egfx_macos_sierra_ug.pdf




 

Late 2013 rMBP
-----------------

* https://everymac.com/systems/apple/macbook_pro/specs/macbook-pro-core-i7-2.0-15-iris-only-late-2013-retina-display-specs.html

SDXC card slot
~~~~~~~~~~~~~~~~

* https://support.apple.com/en-us/HT204384

* SDXC, 4GB to 2TB

Can I install macOS on an SD storage device and use it as a startup volume?

Use Disk Utility to change the default partition table to GUID. Then format the
card to use the Mac OS Extended file format.

* https://everymac.com/systems/apple/macbook_pro/macbook-pro-retina-display-faq/macbook-pro-retina-display-best-sd-card-storage-options-transcend-minidrive.html

* https://www.amazon.com/exec/obidos/ASIN/B00K73NWK0/blueberrypres-20/

Transcend JetDrive Lite 


boot from SD card
~~~~~~~~~~~~~~~~~~~~~

* https://discussions.apple.com/thread/4671181

external boot drive
~~~~~~~~~~~~~~~~~~~~~

https://datarecovery.wondershare.com/macos-sierra/how-to-install-macos-sierra-on-external-hard-drive.html

https://www.imore.com/how-create-bootable-installer-mac-operating-system-high-sierra


Graphics Performance issue with HS
-----------------------------------

* https://forums.macrumors.com/threads/how-is-the-performance-and-battery-life-of-high-sierra-on-the-late-2013-rmbp.2068405/

Compatibility Issue for macOS High Sierra (HS) with NVIDIA GPU CUDA Driver
----------------------------------------------------------------------------

* :google:`macos high sierra macbook pro late 2013 nvidia`
* https://devtalk.nvidia.com/default/topic/1025945/mac-cuda-9-0-driver-fully-compatible-with-macos-high-sierra-10-13-error-quot-update-required-quot-solved-/



::

    This is currently working for me:

    macOS High Sierra 10.13.2
    NVIDIA GeForce GT 750M 2GB

    CUDA Driver Version 387.99
    GPU Driver Version 378.10.10.10.25.102

    BTW, i had to download and install the CUDA driver manually:
    http://www.nvidia.com/object/mac-driver-archive.html



Mavericks, Sys Prefs > CUDA
------------------------------

::

   CUDA 7.5.30 is available

   CUDA Driver Version: 7.0.29
   GPU Driver Version: 8.26.26 310.40.45f01

    

CUDA Driver History
----------------------

::

    CUDA 387.99 driver for MAC Release Date: 12/08/2017   << change in version numbering, latest  
    CUDA 9.0.222 driver for MAC Release Date: 11/02/2017 
    CUDA 9.0.214 driver for MAC Release Date: 10/18/2017 
    CUDA 9.0.197 driver for MAC Release Date: 09/27/2017
    CUDA 8.0.90 driver for MAC Release Date: 07/21/2017 
    CUDA 8.0.83 driver for MAC Release Date: 05/16/2017 
    CUDA 8.0.81 driver for MAC Release Date: 04/11/2017 
    CUDA 8.0.71 driver for MAC Release Date: 03/28/2017
    CUDA 8.0.63 driver for MAC Release Date: 1/27/2017

    CUDA 8.0.57 driver for MAC Release Date: 12/15/2016
    CUDA 8.0.53 driver for MAC Release Date: 11/22/2016
    CUDA 8.0.51 driver for MAC Release Date: 11/2/2016 
    CUDA 8.0.46 driver for MAC Release Date: 10/3/2016 
    CUDA 7.5.30 driver for MAC Release Date: 6/27/2016   << updater suggesting this one
    CUDA 7.5.29 driver for MAC Release Date: 5/17/2016 
    CUDA 7.5.26 driver for MAC Release Date: 3/22/2016 
    CUDA 7.5.25 driver for MAC Release Date: 1/20/2016

    CUDA 7.5.22 driver for MAC Release Date: 12/09/2015 
    CUDA 7.5.21 driver for MAC Release Date: 10/23/2015 
    CUDA 7.5.20 driver for MAC Release Date: 10/01/2015 
    CUDA 7.0.64 driver for MAC Release Date: 08/19/2015 
    CUDA 7.0.61 driver for MAC Release Date: 08/10/2015 
    CUDA 7.0.52 driver for MAC Release Date: 07/02/2015 
    CUDA 7.0.36 driver for MAC Release Date: 04/09/2015 
    CUDA 7.0.35 driver for MAC Release Date: 04/02/2015 
    CUDA 7.0.29 driver for MAC Release Date: 03/18/2015    << am on this one in Mavericks



* https://devtalk.nvidia.com/default/topic/1025945/cuda-setup-and-installation/mac-cuda-9-0-driver-fully-compatible-with-macos-high-sierra-10-13-error-quot-update-required-quot-solved-/5

::

    OH man I finally figured out what worked for me. Daciang, I tried everything
    and had no results. I contacted Nvidia, and they said it was an Apple Issue at
    this point. I got on support with Apple, and they lead me to a working
    solution. 

    I had updated everything NVIDIA and Apple to it's latest updates. I'm on High
    Sierra 10.13.2 (17C89) and my CUDA driver is 387.99 (just released). GPU driver
    version is 378.10.10.10.25.103. Still, I wasn't able to switch over to my
    graphics card and the cuda preferences tab still said "update required".

    Apple had me reset my SMC and NVRAM, and upon rebooting for the second time, IT
    WORKED! I couldn't believe it. 

    It seems too simple:

    https://support.apple.com/kb/HT201295

    then,

    https://support.apple.com/kb/HT204063

    After you've reset the NVRAM, shut down your computer again and then turn it
    on. Everything should work. Good luck!  #66 Posted 12/22/2017 09:59 PM   




SMC (System Management Controller) Reset
------------------------------------------
    
* https://support.apple.com/kb/HT201295


The SMC is responsible for these and other low-level functions on Intel-based Mac computers:
Responding to presses of the power button
Responding to the display lid opening and closing on Mac notebooks
Battery management
Thermal management
Sudden Motion Sensor (SMS)
Ambient light sensing
Keyboard backlighting
Status indicator light (SIL) management
Battery status indicator lights
Selecting an external (instead of internal) video source for some iMac displays


How to reset the SMC on Mac notebooks with non-removable battery
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Choose Apple menu > Shut Down.
2. After your Mac shuts down, press Shift-Control-Option on the left side of the built-in keyboard, 
   then press the power button at the same time. Hold these keys and the power button for 10 seconds. 
   If you have a MacBook Pro with Touch ID, the Touch ID button is also the power button.
3. Release all keys.
4. Press the power button again to turn on your Mac.




NVRAM Reset
--------------


* https://support.apple.com/en-us/HT204063





CUDA Drivers
---------------

* http://www.nvidia.com/object/mac-driver-archive.html
* http://www.nvidia.com/object/macosx-cuda-387.99-driver.html

::

    Version: 387.99
    Release Date: 2017.12.08
    Operating System: Mac OS
    Language: English (U.S.)
    File Size: 35.5 MB


New Release 387.99
CUDA driver update to support CUDA Toolkit 9.0, macOS 10.13.2 and NVIDIA display driver 378.10.10.10.25.102

macOS CUDA driver version format change

The macOS CUDA driver version now uses the format xxx.xx compare to x.x.x 
to be consistent with our Linux and Windows driver version naming convention.

Recommended CUDA version(s):CUDA 9.0
Supported macOS 10.13.x
An alternative method to download the latest CUDA driver is within macOS environment.  
Access the latest driver through System Preferences > Other > CUDA.  Click 'Install CUDA Update'



Old macOS versions
-------------------

* https://www.macworld.co.uk/how-to/mac-software/how-install-download-old-versions-of-mac-os-x-3629363/






EOU
}
macos-dir(){ echo $(local-base)/env/osx/osx-macos ; }
macos-cd(){  cd $(macos-dir); }
macos-mate(){ mate $(macos-dir) ; }
macos-get(){
   local dir=$(dirname $(macos-dir)) &&  mkdir -p $dir && cd $dir

}
