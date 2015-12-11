xcode-src(){      echo xcode/xcode.bash ; }
xcode-source(){   echo ${BASH_SOURCE:-$(env-home)/$(xcode-src)} ; }
xcode-vi(){       vi $(xcode-source) ; }
xcode-env(){      elocal- ; }
xcode-usage(){ cat << EOU

XCODE
======

Updating Xcode
-----------------

Latest
~~~~~~~~~

App Store.app lists only latest Xcode 7.2, which eequires OSX 10.10.5 or later 
(but I am not willing to upgrade currently, still at Mavericks 10.9.4)

Finding which version of Xcode for an OSX version...
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compatibility table https://en.wikipedia.org/wiki/Xcode suggests 
latest Xcode for OSX 10.9.4 is Xcode 6.2 released March 9, 2015 

To get specific versions of Xcode use https://developer.apple.com/downloads/
Safari downloads the ~2GB slowly (~1MB/s) as dmg 

xcode-select
-------------

::

    delta:workflow blyth$ xcode-select -p
    /Applications/Xcode.app/Contents/Developer

    simon:~ blyth$ xcode-select -p
    /Applications/Xcode6-Beta3.app/Contents/Developer


::

    delta:~ blyth$ xcode-select 
    xcode-select: error: no command option given
    Usage: xcode-select [options]

    Print or change the path to the active developer directory. This directory
    controls which tools are used for the Xcode command line tools (for example, 
    xcodebuild) as well as the BSD development commands (such as cc and make).

    Options:
      -h, --help                  print this help message and exit
      -p, --print-path            print the path of the active developer directory
      -s <path>, --switch <path>  set the path for the active developer directory
      -v, --version               print the xcode-select version
      -r, --reset                 reset to the default command line tools path
    delta:~ blyth$ 



commandline tools
-------------------

* http://osxdaily.com/2014/02/12/install-command-line-tools-mac-os-x/

::
 
   xcode-select --install   # apparently allows downloading tools without full Xcode 



xcode underpinnings : SourceKit
--------------------------------

* http://www.jpsim.com/uncovering-sourcekit/

* https://github.com/jpsim/SourceKitten






EOU
}
xcode-dir(){ echo $(env-home)/xcode ; }
xcode-cd(){  cd $(xcode-dir); }

xcode-beta3(){
   echo /Applications/Xcode6-Beta3.app/Contents/Developer
}

xcode-select-cmd(){
  case ${1:-default} in 
    default) echo xcode-select --reset ;;
      beta3) echo xcode-select --switch $(xcode-beta3) ;;
  esac  
}

xcode-select(){
   local cmd=$($FUNCNAME-cmd $1)
   echo sudo $cmd
   sudo $cmd 

   xcrun --show-sdk-path
}

xcode-info(){
   xcrun --show-sdk-path
}

