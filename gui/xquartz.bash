# === func-gen- : gui/xquartz fgp gui/xquartz.bash fgn xquartz fgh gui
xquartz-src(){      echo gui/xquartz.bash ; }
xquartz-source(){   echo ${BASH_SOURCE:-$(env-home)/$(xquartz-src)} ; }
xquartz-vi(){       vi $(xquartz-source) ; }
xquartz-env(){      elocal- ; }
xquartz-usage(){ cat << EOU

XQuartz
========

Progeny of ye olde X11.app, a dependency of some Chroma dependencies

Notes from wikipedia
---------------------

* http://en.wikipedia.org/wiki/XQuartz

As of OS X Mountain Lion, Apple has dropped dedicated support for X11.app, with
users directed to the XQuartz open source project http://support.apple.com/kb/ht5293

As of version 2.7.4, X11.app/XQuartz does not expose support for
high-resolution Retina displays to X11 apps, which run in pixel-doubled mode on
high-resolution displays.


XQuartz 2.7.5 on 10.9.1 Mavericks
----------------------------------

Using "Show files" in the pkg installer, shows the paths that will be written are::

   /Applications/Utilitites/XQuartz.app
   /Library/LaunchAgents/org.macosforge.xquartz.startx.plist
   /Library/LaunchDaemons/org.macosforge.xquartz.privileged_startx.plist
   /opt/X11/
   /private/etc/manpaths.d/
   /private/etc/paths.d/   

Installer states that takes 164.1 MB.


An /Applications/Utilities/X11.app exists but its just a shim that 
displays a message saying that X11 is no longer provided.


Releases
---------

* http://xquartz.macosforge.org/trac/wiki/Releases

::

    X11 2.7.5 - 2013.11.10 - First release supported on Mavericks


EOU
}
xquartz-dir(){ echo $(local-tmp)/env/gui/xquartz ; }
xquartz-cd(){  cd $(xquartz-dir); }
xquartz-mate(){ mate $(xquartz-dir) ; }
xquartz-get(){
   local dir=$(dirname $(xquartz-dir)) &&  mkdir -p $dir && cd $dir
   local url=http://xquartz.macosforge.org/downloads/SL/XQuartz-2.7.5.dmg
   local dmg=$(basename $url)
   [ ! -f $dmg ] && curl -L -O $url

   open $dmg    # Finder GUI pops up with XQuartz.pkg inside 

}
