xcode-src(){      echo xcode/xcode.bash ; }
xcode-source(){   echo ${BASH_SOURCE:-$(env-home)/$(xcode-src)} ; }
xcode-vi(){       vi $(xcode-source) ; }
xcode-env(){      elocal- ; }
xcode-usage(){ cat << \EOU

XCODE
======

See Also
----------

* clt- regarding CommandLineTools, xcode-select, xcrun


::

    Nov 2, 2018 : currently on macOS High Sierra 10.13.4


Xcode 10.1
------------

* https://developer.apple.com/documentation/xcode_release_notes/xcode_10_1_release_notes

Xcode 10.1 includes SDKs for iOS 12.1, watchOS 5.1, macOS 10.14.1, and tvOS
12.1. Xcode 10.1 supports on-device debugging for iOS 8 and later, tvOS 9 and
later, and watchOS 2 and later. Xcode 10.1 requires a Mac running macOS 10.13.6
or later.

Xcode 10
-----------------

* https://developer.apple.com/documentation/xcode_release_notes/xcode_10_release_notes

Xcode 10 is available in the Mac App Store and includes SDKs for iOS 12,
watchOS 5, macOS 10.14, and tvOS 12. Xcode 10 supports on-device debugging for
iOS 8 and later, tvOS 9 and later, and watchOS 2 and later. Xcode 10 requires a
Mac running macOS 10.13.6 or later.



How to handle multiple Xcode ?
---------------------------------

::

    epsilon:Xcode blyth$ ll /Applications/Xcode/
    drwxr-xr-x   3 blyth  staff    96 Mar 23  2018 Xcode_9_2.app



empty app (no storyboard or xib)
----------------------------------

* https://stackoverflow.com/questions/25783282/how-to-create-an-empty-application-in-xcode-without-storyboard


* https://stackoverflow.com/questions/28792722/osx-application-without-storyboard-or-xib-files-using-swift


macOS : main.swift + AppDelegate.swift
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    import Cocoa

    let delegate = AppDelegate()
    NSApplication.shared.delegate = delegate

    NSApplicationMain(CommandLine.argc, CommandLine.unsafeArgv)


Renaming groups breaks the build
-----------------------------------

Renamed the yellow groups, replacing spaces in the names with underscores.
eg renamed "MetalCross2 Shared" -> "MetalCross2_Shared"

* observe xcode 9.3 does git renaming in this circumstance (although my 
  git repo was not setup by xcode).

* build broken, in macOS had to relocate the Info.plist  
* still broken at "precompile bridging header" stage for macOS

A kludge symbolic link fixes this::

   ln -s MetalCross2_Shared MetalCross2\ Shared

But better to avoid such kludge. Search reveals the 
setting thats wrong as SWIFT_OBJC_BRIDGING_HEADER::

    epsilon:MetalCross2 blyth$ find . -type f -exec grep -H ShaderTypes.h {} \;
    ./MetalCross2_Shared/Mandelbrot.metal:#import "ShaderTypes.h"
    ./MetalCross2_Shared/ShaderTypes.h://  ShaderTypes.h
    ./MetalCross2_Shared/ShaderTypes.h:#ifndef ShaderTypes_h
    ./MetalCross2_Shared/ShaderTypes.h:#define ShaderTypes_h
    ./MetalCross2_Shared/ShaderTypes.h:#endif /* ShaderTypes_h */
    ./MetalCross2.xcodeproj/project.pbxproj:		F959CDE2207F6F0F00B7DCF6 /* ShaderTypes.h in Resources */ = {isa = PBXBuildFile; fileRef = F959CDBC207F6F0D00B7DCF6 /* ShaderTypes.h */; };
    ./MetalCross2.xcodeproj/project.pbxproj:		F959CDE3207F6F0F00B7DCF6 /* ShaderTypes.h in Resources */ = {isa = PBXBuildFile; fileRef = F959CDBC207F6F0D00B7DCF6 /* ShaderTypes.h */; };
    ./MetalCross2.xcodeproj/project.pbxproj:		F959CDBC207F6F0D00B7DCF6 /* ShaderTypes.h */ = {isa = PBXFileReference; lastKnownFileType = sourcecode.c.h; path = ShaderTypes.h; sourceTree = "<group>"; };
    ./MetalCross2.xcodeproj/project.pbxproj:				F959CDBC207F6F0D00B7DCF6 /* ShaderTypes.h */,
    ./MetalCross2.xcodeproj/project.pbxproj:				F959CDE2207F6F0F00B7DCF6 /* ShaderTypes.h in Resources */,
    ./MetalCross2.xcodeproj/project.pbxproj:				F959CDE3207F6F0F00B7DCF6 /* ShaderTypes.h in Resources */,
    ./MetalCross2.xcodeproj/project.pbxproj:				SWIFT_OBJC_BRIDGING_HEADER = MetalCross2_Shared/ShaderTypes.h;
    ./MetalCross2.xcodeproj/project.pbxproj:				SWIFT_OBJC_BRIDGING_HEADER = MetalCross2_Shared/ShaderTypes.h;
    ./MetalCross2.xcodeproj/project.pbxproj:				SWIFT_OBJC_BRIDGING_HEADER = "MetalCross2 Shared/ShaderTypes.h";
    ./MetalCross2.xcodeproj/project.pbxproj:				SWIFT_OBJC_BRIDGING_HEADER = "MetalCross2 Shared/ShaderTypes.h";
    Binary file ./MetalCross2.xcodeproj/project.xcworkspace/xcuserdata/blyth.xcuserdatad/UserInterfaceState.xcuserstate matches
    epsilon:MetalCross2 blyth$ 


To find that in settings using the "Levels" view and flip between the iOS and macOS targets
to observe the differences.

For macOS target the  "Swift Compiler - General" > "ObjC generated Interface Header Name"
(aka SWIFT_OBJC_INTERFACE_HEADER_NAME  : use option-click on the title to see the underlying key name)
note that the old "MetalCross2 Shared" name is still in the path.





Vids
-----

* https://developer.apple.com/videos/wwdc2016/


UI Hookup to Code
-------------------

* https://developer.apple.com/library/content/referencelibrary/GettingStarted/DevelopiOSAppsSwift/ConnectTheUIToCode.html


iOS/macOS Cross platform approaches
-------------------------------------

* https://stackoverflow.com/questions/3596455/whats-a-suitable-cross-platform-methodology-for-ios-and-mac-os-x

* https://developer.apple.com/videos/play/wwdc2014/233/

  Sharing code between iOS and macOS


Good Blogs on Xcode/Swift
---------------------------

* https://useyourloaf.com/archives/

* http://mhorga.org/archive/

* http://metalkit.org/ Tutorial series 


Alternative : cmake
---------------------

* https://github.com/sheldonth/ios-cmake


Workspaces vs Projects vs SubProjects
----------------------------------------

* https://developer.apple.com/library/content/featuredarticles/XcodeConcepts/Concept-Workspace.html

The workspace document contains pointers to the included projects and other
files, but no other data. A project can belong to more than one workspace.


Resources from a Framework
----------------------------

* https://useyourloaf.com/blog/loading-resources-from-a-framework/


Playground + Project + Workspace
-----------------------------------

* https://useyourloaf.com/blog/adding-playgrounds-to-xcode-projects/ 

Placing classes into a Framework inside the project, with public access,
allows them to be imported into the playground thats in the same 
workspace.


Code Folding in Xcode 9.3
-----------------------------

* See *Editor > Code Folding* 

* opt-shift-cmd-left Fold all funcs and methods, handy for header like view 

* https://stackoverflow.com/questions/44401963/xcode-9-expand-collapse-braces-brackets-side-bar-missing/45756408

* cmd-click on opening or closing curly, it gets highlighted, 
  hold a while and then can then select fold from menu


Cocoapods vs carthage (emacs vs vim)
----------------------------------------

* https://medium.com/xcblog/carthage-or-cocoapods-that-is-the-question-1074edaafbcb


Swift Package Manager : Package.swift
-----------------------------------------

* reminiscent of setup.py 

::

    swift package help 
    swift package show-dependencies
    swift package resolve
    swift package generate-xcodeproj



Huh : copied Xcode bloated to 17G from original 5.3G  ? 
-----------------------------------------------------------------

Wow compression is having a huge effect ?


    epsilon:home blyth$ diff -r --brief /Volumes/TestHighSierra/Applications/Xcode_9_2.app /Applications/Xcode/Xcode_9_2.app
    Only in /Volumes/TestHighSierra/Applications/Xcode_9_2.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/System/Library/Frameworks/Ruby.framework/Headers/ruby: ruby
    Only in /Volumes/TestHighSierra/Applications/Xcode_9_2.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/System/Library/Frameworks/Ruby.framework/Versions/2.3/Headers/ruby: ruby
    Only in /Volumes/TestHighSierra/Applications/Xcode_9_2.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/System/Library/Frameworks/Ruby.framework/Versions/Current/Headers/ruby: ruby
    Only in /Volumes/TestHighSierra/Applications/Xcode_9_2.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.13.sdk/System/Library/Frameworks/Ruby.framework/Headers/ruby: ruby
    Only in /Volumes/TestHighSierra/Applications/Xcode_9_2.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.13.sdk/System/Library/Frameworks/Ruby.framework/Versions/2.3/Headers/ruby: ruby
    Only in /Volumes/TestHighSierra/Applications/Xcode_9_2.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.13.sdk/System/Library/Frameworks/Ruby.framework/Versions/Current/Headers/ruby: ruby


    epsilon:~ blyth$ l /Applications/Xcode/Xcode_9_2.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/System/Library/Frameworks/Ruby.framework/Headers/ruby/
    total 504
    -rw-r--r--  1 blyth  admin  -  1684 Mar 31 15:18 vm.h
    ...

    epsilon:~ blyth$ l /Volumes/TestHighSierra/Applications/Xcode_9_2.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/System/Library/Frameworks/Ruby.framework/Headers/ruby/
    total 112
    lrwxr-xr-x  1 blyth  staff  -              1 Mar 23 21:02 ruby -> .
    -rw-r--r--  2 blyth  staff  compressed  1684 Jul 16  2017 vm.h
    ...

    * https://superuser.com/questions/382120/mac-os-x-not-reporting-directory-sizes-correctly


Fix by using ditto to copy Xcode.app
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    cd /Applications/Xcode

    mv Xcode_9_2.app Xcode_9_2_bloated.app

    ditto /Volumes/TestHighSierra/Applications/Xcode_9_2.app /Applications/Xcode/Xcode_9_2.app

    epsilon:Xcode blyth$ du -hs /Applications/Xcode/Xcode_9_2.app
    5.3G	/Applications/Xcode/Xcode_9_2.app

    epsilon:Xcode blyth$ du -hs /Volumes/TestHighSierra/Applications/Xcode_9_2.app
    5.3G	/Volumes/TestHighSierra/Applications/Xcode_9_2.app

    epsilon:Xcode blyth$ du -hs Xcode_9_2_bloated.app
     17G	Xcode_9_2_bloated.app

    epsilon:Xcode blyth$ sudo rm -rf Xcode_9_2_bloated.app




xcode clang version sequence
-------------------------------

* https://trac.macports.org/wiki/XcodeVersionInfo

::

    ## following xcode-;xcode-92 (93 is default)

    epsilon:~ blyth$ /usr/bin/xcodebuild -version
    Xcode 9.2
    Build version 9C40b

    epsilon:~ blyth$ clang --version
    Apple LLVM version 9.0.0 (clang-900.0.39.2)
    Target: x86_64-apple-darwin17.5.0
    Thread model: posix
    InstalledDir: /Applications/Xcode/Xcode_9_2.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin

 

cannot run a playground in debugger !
----------------------------------------

* this makes them fit for snippets only 
* https://stackoverflow.com/questions/34186805/swift-playground-with-debugger-support


xcodeproj into repo ?
------------------------

* https://cocoacasts.com/setting-up-a-brand-new-project-in-xcode


* http://shanesbrain.net/2008/7/9/using-xcode-with-git


.gitattributes to treat as binary::

    *.pbxproj -crlf -diff -merge

The line in .gitattributes treats your Xcode project file as a binary. This
prevents Git from trying to fix newlines, show it in diffs, and excludes it
from merges. Note that you will still see it shown as a conflict in merges,
although the file won't have changed. Simply commit it and things should be
good.


* :google:`xcodeproj gitattributes`

* https://robots.thoughtbot.com/xcode-and-git-bridging-the-gap

::

   *.pbxproj binary merge=union


* https://stackoverflow.com/questions/8026429/should-i-git-ignore-xcodeproject-project-pbxproj-file
* https://github.com/github/gitignore/blob/master/Global/Xcode.gitignore
* https://peterwitham.com/swift-archives/create-a-gitignore-for-swift/


Preserving Xcode 9.2
-----------------------

::

    cp -r /Volumes/TestHighSierra/Applications/Xcode_9_2.app /Applications/Xcode/

    cp: /Volumes/TestHighSierra/Applications/Xcode_9_2.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/System/Library/Frameworks/Ruby.framework/Headers/ruby/ruby: directory causes a cycle
    cp: /Volumes/TestHighSierra/Applications/Xcode_9_2.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/System/Library/Frameworks/Ruby.framework/Versions/2.3/Headers/ruby/ruby: directory causes a cycle
    cp: /Volumes/TestHighSierra/Applications/Xcode_9_2.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/System/Library/Frameworks/Ruby.framework/Versions/Current/Headers/ruby/ruby: directory causes a cycle
    cp: /Volumes/TestHighSierra/Applications/Xcode_9_2.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.13.sdk/System/Library/Frameworks/Ruby.framework/Headers/ruby/ruby: directory causes a cycle
    cp: /Volumes/TestHighSierra/Applications/Xcode_9_2.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.13.sdk/System/Library/Frameworks/Ruby.framework/Versions/2.3/Headers/ruby/ruby: directory causes a cycle
    cp: /Volumes/TestHighSierra/Applications/Xcode_9_2.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.13.sdk/System/Library/Frameworks/Ruby.framework/Versions/Current/Headers/ruby/ruby: directory causes a cycle
    epsilon:~ blyth$ 



Multiple Versions of Xcode and CommandLineTools ?
----------------------------------------------------

* https://medium.com/@hacknicity/working-with-multiple-versions-of-xcode-e331c01aa6bc

* :google:`Multiple Versions of Xcode and Mac App Store Updates`

* http://iosbrain.com/blog/2017/01/02/installing-multiple-versions-of-xcode-6-7-8-side-by-side-together-on-the-same-mac/

* https://stackoverflow.com/questions/669367/can-i-have-multiple-xcode-versions-installed

iosdevelopertips (2014)
~~~~~~~~~~~~~~~~~~~~~~~~~

* http://iosdevelopertips.com/xcode/install-multiple-versions-of-xcode.html

Prior to the Mac App Store, it was quite simple to work with multiple versions
of Xcode, as each release had an option to specify the destination folder
during install. No more, installations (non-beta) will overwrite an installed
version of an app with the latest release, as each new release is installed in
the /Applications folder with the name Xcode.app.



Updating Xcode
-----------------



Mar 31, 2018 TestHighSierra with Xcode 9.2
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CUDA release notes say that latest 9.smth works with Xcode 9.2 
but on Epsilon already upgraded to Xcode 9.3.  Seems that 
should always rename /Applications/Xcode.app 
to eg /Applications/Xcode_9.2_9C40b.app


(Mar 30, 2018) Epsilon High Sierra, with latest Xcode misses /usr/include
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://superuser.com/questions/995360/missing-usr-include-in-os-x-el-capitan

Suggests to run *xcode-select --install* instead of doing this
tried opening Xcode.app (previously unopened) which started out "Installing Components..."
But even after that had to::

    delta:~ blyth$ xcode-select --install
    xcode-select: note: install requested for command line developer tools
    delta:~ blyth$ 
    delta:~ blyth$ 
    delta:~ blyth$ ll /usr/include/
    total 1712
    drwxr-xr-x    5 root  wheel      160 Oct  7 09:28 architecture
    -rw-r--r--    1 root  wheel     3139 Oct  7 09:35 setjmp.h
    -rw-r--r--    1 root  wheel     2054 Oct  7 09:35 ucontext.h


Latest (Jan 2018) Version : Xcode 9.2 (includes Swift 4)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compatibility : macOS 10.12.6 or later


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


Installing Xcode 6.2
~~~~~~~~~~~~~~~~~~~~~~

::

   open ~/Downloads/Xcode_6.2.dmg 

   # suggests to drag Xcode.app to /Applications/
   # do so after renaming old Xcode.app to Xcode-511.app   
   # 2.5G dmg decompressed to 5.8G


::

    simon:~ blyth$ clang --version
    Apple LLVM version 6.0 (clang-600.0.57) (based on LLVM 3.5svn)
    Target: x86_64-apple-darwin13.3.0
    Thread model: posix


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

xcode-v(){  
  local app=${1:-/Applications/Xcode.app}
  [ ! -d "$app" ] && echo $msg no app $app && return
  local pls="$app/Contents/Info.plist"
  [ ! -f "$pls" ] && echo $msg no pls $pls && return
  local ver=$(/usr/libexec/PlistBuddy -c "Print CFBundleShortVersionString" $pls)

  echo $msg $pls $ver  
}

xcode-beta3(){
   echo /Applications/Xcode6-Beta3.app/Contents/Developer
}

xcode-select-cmd(){
  case ${1:-default} in 
    default) echo xcode-select- --reset ;;
      beta3) echo xcode-select- --switch $(xcode-beta3) ;;
  esac  
}


xcode-switch-notes(){ cat << EON

::

   -s <path>, --switch <path>
          Sets the active developer directory to the given path, for example
          /Applications/Xcode-beta.app. This command must be run with superuser
          permissions (see sudo(8)),  and will  affect  all  users  on the system. To set
          the path without superuser permissions or only for the current shell session,
          use the DEVELOPER_DIR environment variable instead (see ENVIRONMENT).

   -p, --print-path
          Prints the path to the currently selected developer directory. This
          is useful for inspection, but scripts and other tools should use xcrun(1) to
          locate tool inside  the active developer directory.

   -r, --reset
          Unsets  any  user-specified developer directory, so that the
          developer directory will be found via the default search mechanism. This
          command must be run with superuser permissions (see sudo(8)), and will affect
          all users on the system.


EON
}


xcode-cl(){ xcode-switch- /Library/Developer/CommandLineTools ; }
xcode-93(){ xcode-switch /Applications/Xcode.app ; }
xcode-92(){ xcode-switch /Applications/Xcode/Xcode_9_2.app ; }

xcode-switch(){
   local app=${1:-/Applications/Xcode.app}
   local devl=$app/Contents/Developer
   xcode-switch- $devl
}

xcode-switch-(){
   local dir=$1
   local msg="=== $FUNCNAME :"
   [ ! -d "$dir" ] && echo $msg missing dir $dir  && return 
   local cmd="sudo xcode-select --switch $dir"
   echo $cmd
   eval $cmd
}





xcode-check(){ $FUNCNAME- | -xcode-do ; }
xcode-check-(){ cat << EOC
xcode-select --print-path

which clang 
clang --version

which git
git --version

xcrun -find clang
xcrun -find git

xcrun --show-sdk-path
xcrun --show-sdk-version 
xcrun --show-sdk-build-version
xcrun --show-sdk-platform-version

EOC
}

-xcode-do(){
  local cmd
  local rc
  while read cmd ; do 
     echo $cmd 
     [ "${cmd:0:1}" == "#" -o "${cmd:0:1}" == " " ] && continue
     eval $cmd
     rc=$?
     [ "$rc" != "0" ] && echo NON-ZERO-RC $RC && return 
  done
}


xcode-tools-pkgs-(){ pkgutil --pkgs | grep -i tools  ; }
xcode-tools-pkgs(){
   local pkg
   local cmd
   $FUNCNAME- | while read pkg ; do 
       cmd="pkgutil --pkg-info=$pkg"
       echo
       echo $cmd
       eval $cmd
   done
}

xcode-switch-test-notes(){ cat << EON


Test Switching between xcode-92 xcode-93 and xcode-cl
========================================================

Command line Tools Version
---------------------------

* https://apple.stackexchange.com/questions/180957/determine-xcode-command-line-tools-version

::

    epsilon:~ blyth$ pkgutil --pkgs | grep -i tools 
    com.apple.pkg.CLTools_Executables
    com.apple.pkg.CLTools_SDK_OSX1012
    com.apple.pkg.CLTools_SDK_macOSSDK
    com.apple.pkg.CLTools_SDK_macOS1013

    epsilon:~ blyth$ pkgutil --pkg-info=com.apple.pkg.CLTools_Executables
    package-id: com.apple.pkg.CLTools_Executables
    version: 9.3.0.0.1.1521514116
    volume: /
    location: /
    install-time: 1522422695
    groups: com.apple.FindSystemFiles.pkg-group 
    epsilon:~ blyth$ 

    epsilon:~ blyth$ date -r 1522422695
    Fri Mar 30 23:11:35 CST 2018

    epsilon:~ blyth$ date -r 1521514116
    Tue Mar 20 10:48:36 CST 2018

    epsilon:~ blyth$ xcode-;xcode-tools-pkgs | grep version
    version: 9.3.0.0.1.1521514116
    version: 9.3.0.0.1.1521514116
    version: 9.3.0.0.1.1521514116
    version: 9.3.0.0.1.1521514116
    epsilon:~ blyth$ 
    epsilon:~ blyth$ xcode-;xcode-tools-pkgs | grep time
    install-time: 1522422695
    install-time: 1522422696
    install-time: 1522422695
    install-time: 1522422696




Hmm : what about the headers ?
---------------------------------

* :google:`does xcode-select switch /usr/include headers ?`

Easy to play shim tricks with binaries, but what about the 
headers in /usr/include ? This might be a theoretical problem only, 
as these "system" headers are hopefully very mature.


Some clarification from NVIDIA on Command-Line Tools
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://developer.download.nvidia.com/compute/cuda/9.1/Prod/docs/sidebar/CUDA_Installation_Guide_Mac.pdf

The CUDA Toolkit requires that the native command-line tools are already installed on
the system. Xcode must be installed before these command-line tools can be installed.
The command-line tools can be installed by running the following command::

    xcode-select --install

Note: **It is recommended to re-run the above command if Xcode is upgraded, 
or an older version of Xcode is selected**.

You can verify that the toolchain is installed by running the following command::

    /usr/bin/cc --version


BUT::

    epsilon:~ blyth$ xcode-select --install
    xcode-select: error: command line tools are already installed, use "Software Update" to install updates





xcode-92 : Apple LLVM version 9.0.0 (clang-900.0.39.2)
--------------------------------------------------------

::

    epsilon:env blyth$ xcode-;xcode-92
    sudo xcode-select --switch /Applications/Xcode/Xcode_9_2.app/Contents/Developer

    epsilon:env blyth$ xcode-;xcode-check
    xcode-select --print-path
    /Applications/Xcode/Xcode_9_2.app/Contents/Developer

    which clang
    /usr/bin/clang
    clang --version
    Apple LLVM version 9.0.0 (clang-900.0.39.2)
    Target: x86_64-apple-darwin17.5.0
    Thread model: posix
    InstalledDir: /Applications/Xcode/Xcode_9_2.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin

    which git
    /usr/bin/git
    git --version
    git version 2.14.3 (Apple Git-98)

    xcrun --show-sdk-path
    /Applications/Xcode/Xcode_9_2.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk
    xcrun --show-sdk-version
    10.13
    xcrun --show-sdk-build-version
    17C76
    xcrun --show-sdk-platform-version
    1.1


xcode-93 : Apple LLVM version 9.1.0 (clang-902.0.39.1)
--------------------------------------------------------

::

    epsilon:env blyth$ xcode-;xcode-93
    sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer

    epsilon:env blyth$ xcode-;xcode-check
    xcode-select --print-path
    /Applications/Xcode.app/Contents/Developer

    which clang
    /usr/bin/clang
    clang --version
    Apple LLVM version 9.1.0 (clang-902.0.39.1)
    Target: x86_64-apple-darwin17.5.0
    Thread model: posix
    InstalledDir: /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin

    which git
    /usr/bin/git
    git --version
    git version 2.15.1 (Apple Git-101)

    xcrun --show-sdk-path
    /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.13.sdk
    xcrun --show-sdk-version
    10.13
    xcrun --show-sdk-build-version
    17E189
    xcrun --show-sdk-platform-version
    1.1



xcode-cl : versions of git and clang match xcode-93
-----------------------------------------------------

::

    epsilon:env blyth$ xcode-;xcode-cl
    sudo xcode-select --switch /Library/Developer/CommandLineTools

    epsilon:env blyth$ xcode-;xcode-check
    xcode-select --print-path
    /Library/Developer/CommandLineTools

    which clang
    /usr/bin/clang
    clang --version
    Apple LLVM version 9.1.0 (clang-902.0.39.1)
    Target: x86_64-apple-darwin17.5.0
    Thread model: posix
    InstalledDir: /Library/Developer/CommandLineTools/usr/bin

    which git
    /usr/bin/git
    git --version
    git version 2.15.1 (Apple Git-101)

    xcrun -find clang
    /Library/Developer/CommandLineTools/usr/bin/clang

    xcrun -find git
    /Library/Developer/CommandLineTools/usr/bin/git

    xcrun --show-sdk-path
    /Library/Developer/CommandLineTools/SDKs/MacOSX10.13.sdk
    xcrun --show-sdk-version
    10.13.4
    xcrun --show-sdk-build-version
    17E189
    xcrun --show-sdk-platform-version
    xcrun: error: unable to lookup item 'PlatformVersion' from command line tools installation
    xcrun: error: unable to lookup item 'PlatformVersion' in SDK '/'
    NON-ZERO-RC
    epsilon:env blyth$ 



hmm versions of command lines tools from TestHighSierra are ahead of the Xcode 92 from there ?
-------------------------------------------------------------------------------------------------

* this means am missing the command lines tools corresponding to xcode-92

::

    epsilon:env blyth$ xcode-;xcode-switch- /Volumes/TestHighSierra/Library/Developer/CommandLineTools/
    sudo xcode-select --switch /Volumes/TestHighSierra/Library/Developer/CommandLineTools/
    Password:
    epsilon:env blyth$ xcode-;xcode-check 
    xcode-select --print-path
    /Volumes/TestHighSierra/Library/Developer/CommandLineTools

    which clang
    /usr/bin/clang
    clang --version
    Apple LLVM version 9.1.0 (clang-902.0.39.1)
    Target: x86_64-apple-darwin17.5.0
    Thread model: posix
    InstalledDir: /Volumes/TestHighSierra/Library/Developer/CommandLineTools/usr/bin

    which git
    /usr/bin/git
    git --version
    git version 2.15.1 (Apple Git-101)

    xcrun -find clang
    /Volumes/TestHighSierra/Library/Developer/CommandLineTools/usr/bin/clang
    xcrun -find git
    /Volumes/TestHighSierra/Library/Developer/CommandLineTools/usr/bin/git

    xcrun --show-sdk-path
    /Volumes/TestHighSierra/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk
    xcrun --show-sdk-version
    10.13.4
    xcrun --show-sdk-build-version
    17E189
    xcrun --show-sdk-platform-version
    xcrun: error: unable to lookup item 'PlatformVersion' from command line tools installation
    xcrun: error: unable to lookup item 'PlatformVersion' in SDK '/'
    NON-ZERO-RC
    epsilon:env blyth$ 



EON
}

 



