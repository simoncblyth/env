clt-source(){   echo $BASH_SOURCE ; }
clt-vi(){       vi $(clt-source) $(xcode-source) ; }
clt-env(){      elocal- ; xcode- ;  }
clt-usage(){ cat << \EOU

xcode : Command Line Tools
============================

::

   epsilon:~ blyth$ ll /Applications/Xcode/Xcode_9_2.app/Contents/Developer/usr/bin/ | wc -l 
         104
    epsilon:~ blyth$ ll /Applications/Xcode.app/Contents/Developer/usr/bin/ | wc -l 
         103
    epsilon:~ blyth$ ll /usr/bin/ | wc -l 
         979


::

    In [1]: a=set(os.listdir("/Applications/Xcode.app/Contents/Developer/usr/bin"))

    In [2]: b=set(os.listdir("/usr/bin"))

    In [3]: c=set(os.listdir("/Applications/Xcode/Xcode_9_2.app/Contents/Developer/usr/bin"))

    In [4]: len(a)
    Out[4]: 100

    In [5]: len(b)
    Out[5]: 976

    In [6]: len(c)
    Out[6]: 101

    In [7]: c - a  # 92 but not 93 ... dropped
    Out[7]: {'docsetutil', 'iprofiler'}

    In [17]: a - c    # in 93 but not 92 ... added
    Out[17]: {'xccov'}


    In [10]: len(a-b)   # binaries in 93 but not /usr/bin
    Out[10]: 35

    In [11]: len(c-b)   # in 92 but not /usr/bin
    Out[11]: 35

    In [15]: (a-b)-(c-b)
    Out[15]: {'xccov'}

    In [16]: (c-b)-(a-b)
    Out[16]: {'docsetutil'}


    In [8]: a - b
    Out[8]: 
    {'ImageUnitAnalyzer',
     'TextureAtlas',
     'actool',
     'amlint',
     'bitcode-build-tool',
     'convertRichTextToAscii',
     'copySceneKitAssets',
     'copypng',
     'coremlc',
     'extractLocStrings',
     'ibtool3',
      ..



docs
------

* https://developer.apple.com/library/content/technotes/tn2339/_index.html

What is the Command Line Tools Package?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Command Line Tools Package is a small self-contained package available for
download separately from Xcode and that allows you to do command line
development in macOS. It consists of the macOS SDK and command-line tools such
as Clang, which are installed in the /Library/Developer/CommandLineTools
directory.


Downloading command-line tools is not available in Xcode for macOS 10.9. How can I install them on my machine?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Install Xcode**

If Xcode is installed on your machine, then there is no need to install them.
Xcode comes bundled with all your command-line tools. macOS 10.9 and later
includes shims or wrapper executables. These shims, installed in /usr/bin, can
map any tool included in /usr/bin to the corresponding one inside Xcode. xcrun
is one of such shims, which allows you to find or run any tool inside Xcode
from the command line. Use it to invoke any tool within Xcode from the command
line as shown in Listing 1.

::

    xcrun dwarfdump --uuid  MySample.app/MySample


**Download the Command Line Tools package from the Developer website**

The Command Line Tools package is available for download on the Download for
Apple Developers page. Log in with your Apple ID, then search and download the
Command Line Tools package appropriate for your machine such as macOS 10.12 as
shown in Figure 1.

* distributed as .dmg 

* Note: In macOS 10.9 and later, Software update notifies you when new versions
  of the command-line tools are available for update.


**Install the Command Line Tools package via the Terminal application**

You can install the Command Line Tools package by running the xcode-select --install command.

* Note: macOS comes bundled with xcode-select, a command-line tool that is
  installed in /usr/bin. It allows you to manage the active developer directory
  for Xcode and other BSD development tools. See its man page for more
  information.


How can I uninstall the command-line tools?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Xcode includes all of the command-line tools. If it is installed on your
  system, remove it to uninstall the command-line tools.

* If the /Library/Developer/CommandLineTools directory exists on your system,
  remove it to uninstall the command-line tools.





seems /Library/Developer/CommandLineTools was formerly an alternative to full Xcode.app
-----------------------------------------------------------------------------------------

Did migration assistant make a mess in /usr ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    epsilon:~ blyth$ ll /Volumes/TestHighSierra/usr/bin/ | wc -l 
         977
    epsilon:~ blyth$ ll /Volumes/Delta/usr/bin/ | wc -l 
        1160
    epsilon:~ blyth$ ll /Volumes/Epsilon/usr/bin/ | wc -l 
         979
    epsilon:~ blyth$ ll /usr/bin/ | wc -l 
         979


Early 2015
~~~~~~~~~~~~~~

* https://github.com/nodejs/node-gyp/issues/569

::

    xcode-select --install # Install Command Line Tools if you haven't already.
    sudo xcode-select --switch /Library/Developer/CommandLineTools # Enable command line tools

::

    epsilon:~ blyth$ diff -r --brief /Volumes/TestHighSierra/Library/Developer/CommandLineTools/ /Volumes/Epsilon/Library/Developer/CommandLineTools/
    diff: /Volumes/TestHighSierra/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/System/Library/Frameworks/Ruby.framework/Headers/ruby/ruby: recursive directory loop
    diff: /Volumes/TestHighSierra/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/System/Library/Frameworks/Ruby.framework/Versions/2.3/Headers/ruby/ruby: recursive directory loop
    diff: /Volumes/TestHighSierra/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/System/Library/Frameworks/Ruby.framework/Versions/Current/Headers/ruby/ruby: recursive directory loop
    diff: /Volumes/TestHighSierra/Library/Developer/CommandLineTools/SDKs/MacOSX10.13.sdk/System/Library/Frameworks/Ruby.framework/Headers/ruby/ruby: recursive directory loop
    diff: /Volumes/TestHighSierra/Library/Developer/CommandLineTools/SDKs/MacOSX10.13.sdk/System/Library/Frameworks/Ruby.framework/Versions/2.3/Headers/ruby/ruby: recursive directory loop
    diff: /Volumes/TestHighSierra/Library/Developer/CommandLineTools/SDKs/MacOSX10.13.sdk/System/Library/Frameworks/Ruby.framework/Versions/Current/Headers/ruby/ruby: recursive directory loop
    epsilon:~ blyth$ 



::

    epsilon:~ blyth$ ll /Library/Developer/CommandLineTools/usr/
    total 0
    drwxr-xr-x    7 root  admin   224 Mar 20 11:29 .
    drwxr-xr-x    8 root  admin   256 Mar 20 11:29 share
    drwxr-xr-x    5 root  admin   160 Mar 20 11:30 include
    drwxr-xr-x    6 root  admin   192 Mar 20 11:30 libexec
    drwxr-xr-x   45 root  admin  1440 Mar 30 23:10 lib
    drwxr-xr-x  119 root  admin  3808 Mar 30 23:10 bin
    drwxr-xr-x    5 root  admin   160 Mar 30 23:11 ..
    epsilon:~ blyth$ 

    epsilon:~ blyth$ ll /Volumes/Epsilon/Applications/Xcode.app/Contents/Developer/usr/
    total 0
    drwxr-xr-x    3 root  wheel    96 Feb 14 10:02 libexec
    drwxr-xr-x    6 root  wheel   192 Mar 20 11:27 .
    drwxr-xr-x    9 root  wheel   288 Mar 30 17:43 ..
    drwxr-xr-x  102 root  wheel  3264 Mar 30 17:43 bin
    drwxr-xr-x   42 root  wheel  1344 Mar 30 17:43 lib
    drwxr-xr-x    9 root  wheel   288 Mar 30 17:43 share
    epsilon:~ blyth$ 

    epsilon:~ blyth$ ll /usr/
    total 0
    lrwxr-xr-x    1 root  wheel      8 Jan 15  2014 X11 -> /opt/X11
    drwxr-xr-x    5 root  wheel    160 Jan 19 15:35 standalone
    drwxr-xr-x  248 root  wheel   7936 Mar 30 18:17 sbin
    drwxr-xr-x   10 root  wheel    320 Mar 30 22:08 local
    drwxr-xr-x  978 root  wheel  31296 Mar 30 22:56 bin
    drwxr-xr-x  238 root  wheel   7616 Mar 30 22:56 libexec
    drwxr-xr-x  267 root  wheel   8544 Mar 30 23:11 include
    drwxr-xr-x@  11 root  wheel    352 Mar 30 23:11 .
    drwxr-xr-x  312 root  wheel   9984 Mar 30 23:11 lib
    drwxr-xr-x   47 root  wheel   1504 Mar 30 23:11 share
    drwxr-xr-x   34 root  wheel   1088 Mar 31 09:35 ..
    epsilon:~ blyth$ 






xcode-select : Manages the active developer directory for Xcode and BSD tools.
--------------------------------------------------------------------------------

* x-man-page:://xcode-select

When multiple Xcode applications are installed on a system 
(e.g.  /Applications/Xcode.app, containing the latest Xcode,   
and /Applications/Xcode-beta.app containing a beta) use::

   xcode-select --switch path/to/Xcode.app 

to specify the Xcode that you wish to use for command line developer tools.

After setting a developer directory, all of the xcode-select provided developer tool shims (see FILES) will
automatically  invoke  the  version  of the tool inside the selected developer directory. Your own scripts,
makefiles, and other tools can also use xcrun(1) to easily lookup tools inside the active developer  direc-
tory,  making  it  easy  to  switch them between different versions of the Xcode tools and allowing them to
function properly on systems where the Xcode application has been installed to a non-default location.


xcrun
------

::

    testepsilon:~ blyth$ xcrun 
    Usage: xcrun [options] <tool name> ... arguments ...

    Find and execute the named command line tool from the active developer
    directory.

    The active developer directory can be set using `xcode-select`, or via the
    DEVELOPER_DIR environment variable. See the xcrun and xcode-select manual
    pages for more information.

    Options:
      -h, --help                  show this help message and exit
      --version                   show the xcrun version
      -v, --verbose               show verbose logging output
      --sdk <sdk name>            find the tool for the given SDK name
      --toolchain <name>          find the tool for the given toolchain
      -l, --log                   show commands to be executed (with --run)
      -f, --find                  only find and print the tool path
      -r, --run                   find and execute the tool (the default behavior)
      -n, --no-cache              do not use the lookup cache
      -k, --kill-cache            invalidate all existing cache entries
      --show-sdk-path             show selected SDK install path
      --show-sdk-version          show selected SDK version
      --show-sdk-build-version    show selected SDK build version
      --show-sdk-platform-path    show selected SDK platform path
      --show-sdk-platform-version show selected SDK platform version
    testepsilon:~ blyth$ 


Issue from renaming /Applications/Xcode.app
---------------------------------------------

After renaming /Applications/Xcode.app to /Applications/Xcode_9.2_9C40b.app get error::

    testepsilon:~ blyth$ xcodebuild -showsdks 
    xcode-select: error: tool 'xcodebuild' requires Xcode, but active developer directory '/Library/Developer/CommandLineTools' is a command line tools instance

This error implies that the CommandLineTools is a separate alternative to the tools within the Xcode.app/Contents/Developer, 
see clt-


Adding Xcode.app symbolic link avoids the error::

    testepsilon:Applications blyth$ ln -s Xcode_9.2_9C40b.app Xcode.app

    testepsilon:~ blyth$ xcodebuild -showsdks 
    iOS SDKs:
        iOS 11.2                      	-sdk iphoneos11.2

    iOS Simulator SDKs:
        Simulator - iOS 11.2          	-sdk iphonesimulator11.2

    macOS SDKs:
        macOS 10.13                   	-sdk macosx10.13

    tvOS SDKs:
        tvOS 11.2                     	-sdk appletvos11.2

    tvOS Simulator SDKs:
        Simulator - tvOS 11.2         	-sdk appletvsimulator11.2

    watchOS SDKs:
        watchOS 4.2                   	-sdk watchos4.2

    watchOS Simulator SDKs:
        Simulator - watchOS 4.2       	-sdk watchsimulator4.2

    testepsilon:~ blyth$ 



::

    testepsilon:~ blyth$ xcodebuild -showsdks 
    xcode-select: error: tool 'xcodebuild' requires Xcode, but active developer directory '/Library/Developer/CommandLineTools' is a command line tools instance



Booted into TestHighSierra 10.13.3 with Xcode 9.2
---------------------------------------------------

Running *xcode-select --install* performed download, yielding /Library/Developer/CommandLineTools::

    testepsilon:~ blyth$ ll /Library/Developer/
    total 0
    0 drwxr-xr-x   4 root  admin   136 Mar 31 09:40 .
    0 drwxr-xr-x   5 root  admin   170 Mar 31 09:39 CommandLineTools
    0 drwxr-xr-x+ 61 root  wheel  2074 Mar 23 21:06 ..
    0 drwxr-xr-x   3 root  admin   102 Nov  1 23:54 PrivateFrameworks

    testepsilon:~ blyth$ ll /Library/Developer/CommandLineTools/
    total 0
    0 drwxr-xr-x  4 root  admin  136 Mar 31 09:40 ..
    0 drwxr-xr-x  4 root  wheel  136 Mar 31 09:40 SDKs
    0 drwxr-xr-x  5 root  admin  170 Mar 31 09:39 .
    0 drwxr-xr-x  5 root  admin  170 Mar 20 11:29 Library
    0 drwxr-xr-x  7 root  admin  238 Mar 20 11:29 usr

    testepsilon:~ blyth$ ll /Library/Developer/CommandLineTools/SDKs/
    total 8
    0 drwxr-xr-x  5 root  wheel  170 May 22  2022 MacOSX.sdk
    0 drwxr-xr-x  4 root  wheel  136 Mar 31 09:40 .
    8 lrwxr-xr-x  1 root  wheel   10 Mar 31 09:40 MacOSX10.13.sdk -> MacOSX.sdk
    0 drwxr-xr-x  5 root  admin  170 Mar 31 09:39 ..

    testepsilon:~ blyth$ ll /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/
    total 0
    0 drwxr-xr-x  5 root  wheel   170 May 22  2022 .
    0 drwxr-xr-x  4 root  wheel   136 Mar 31 09:40 ..
    0 drwxr-xr-x  6 root  wheel   204 Mar 15 18:40 usr
    0 -rw-r--r--  1 root  wheel  1260 Mar 15 12:51 SDKSettings.plist
    0 drwxr-xr-x  3 root  wheel   102 Feb 26 14:23 System
    testepsilon:~ blyth$ 


::

    testepsilon:~ blyth$ /usr/bin/xcrun --find xcodebuild
    /Applications/Xcode.app/Contents/Developer/usr/bin/xcodebuild
    testepsilon:~ blyth$ /usr/bin/xcrun --find git
    /Applications/Xcode.app/Contents/Developer/usr/bin/git
    testepsilon:~ blyth$ 
    testepsilon:~ blyth$ 
    testepsilon:~ blyth$ /usr/bin/xcrun --find clang
    /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang
    testepsilon:~ blyth$ 
    testepsilon:~ blyth$ /usr/bin/xcrun --find xcrun
    xcrun: error: unable to find utility "xcrun", not a developer tool or in PATH
    testepsilon:~ blyth$ 






Command Line Tools : how multiple versions are handled
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Initally thought that the tools in their traditional locations /usr/bin/git
might be hardlinks to inodes matching some within /Applications/XCode.app 
But seems not.

With inodes::

    delta:~ blyth$ ls -li /usr/bin/clang /usr/bin/git
    1853161 -rwxr-xr-x  1 root  wheel  18288 Mar 28 12:02 /usr/bin/clang
    1853293 -rwxr-xr-x  1 root  wheel  18288 Mar 28 12:02 /usr/bin/git


    delta:~ blyth$ otool -L /usr/bin/git
    /usr/bin/git:
        /usr/lib/libxcselect.dylib (compatibility version 1.0.0, current version 1.0.0)
        /usr/lib/libSystem.B.dylib (compatibility version 1.0.0, current version 1252.0.0)
    delta:~ blyth$ otool -L /usr/bin/clang
    /usr/bin/clang:
        /usr/lib/libxcselect.dylib (compatibility version 1.0.0, current version 1.0.0)
        /usr/lib/libSystem.B.dylib (compatibility version 1.0.0, current version 1252.0.0)
    delta:~ blyth$ 


    delta:~ blyth$ man find
    delta:~ blyth$ find /Applications/Xcode.app/ -samefile /usr/bin/git
    delta:~ blyth$  ## nope 

::

    delta:~ blyth$ ll -i  /Library/Developer/CommandLineTools/usr/bin/git /Library/Developer/CommandLineTools/usr/bin/clang
    3062589 -rwxr-xr-x  1 root  admin   2056368 Mar 20 11:30 /Library/Developer/CommandLineTools/usr/bin/git
    3062558 -rwxr-xr-x  1 root  admin  73256080 Mar 20 11:30 /Library/Developer/CommandLineTools/usr/bin/clang







EOU
}
clt-dir(){ echo $(local-base)/env/xcode/commandlinetools ; }
clt-cd(){  cd $(clt-dir); }
