# === func-gen- : osx/osx fgp osx/osx.bash fgn osx fgh osx
osx_src(){      echo osx/osx.bash ; }
osx_source(){   echo ${BASH_SOURCE:-$ENV_HOME/$(osx_src)} ; }
osx_vi(){       vi $(osx_source) ; }
osx_env(){      elocal- ; }
osx_usage(){ cat << EOU


screencapture
----------------

::

    screencapture -i -w -W -o -S /tmp/cap.png

A pyvista window of size  [2560, 1440] yields a capture with the window bar chrome of 44px 
2560x1484


SCREENCAPTURE(1)          BSD General Commands Manual         SCREENCAPTURE(1)

NAME
     screencapture -- capture images from the screen and save them to a file or the clipboard

SYNOPSIS
     screencapture [-SWCTMPcimswxto] file

DESCRIPTION
     The screencapture utility is not very well documented to date.  A list of options follows.

     -c      Force screen capture to go to the clipboard.

     -C      Capture the cursor as well as the screen.  Only allowed in non-interactive modes.

     -i      Capture screen interactively, by selection or window.  The control key will cause the screen shot to go to the clipboard.  The space key will toggle between mouse
             selection and window selection modes.  The escape key will cancel the interactive screen shot.

     -m      Only capture the main monitor, undefined if -i is set.

     -M      Open the taken picture in a new Mail message.

     -o      In window capture mode, do not capture the shadow of the window.

     -P      Open the taken picture in a Preview window.

     -s      Only allow mouse selection mode.

     -S      In window capture mode, capture the screen instead of the window.

     -t      <format> Image format to create, default is png (other options include pdf, jpg, tiff and other formats).

     -T      <seconds> Take the picture after a delay of <seconds>, default is 5.

     -w      Only allow window selection mode.

     -W      Start interaction in window selection mode.

     -x      Do not play sounds.

     -a      Do not capture attached windows.

     -r      Do not add screen dpi meta data to captured file.

     -b      capture Touch Bar, only works in non-interactive modes.

     files   where to save the screen capture, 1 file per screen




create new user "francis" for testing g4_1070
------------------------------------------------

* System Preferences > Users & Groups
* add new user name and set passwould 
* login using the GUI, click thru the dialogs skipping things like appleid and Siri 

* make basic GUI customizations:

  * Trackpad > Tap to click 
  * Accessibility > Mouse & Trackpad > Trackpad Options : Enable Dragging (without drag lock)
  * Dock > Autohide

* GUI logout 

* back to main blyth account, add username to .ssh/config
* attempt to ssh in fails::

    epsilon:notes blyth$ ssh F
    Password:
    Connection closed by 127.0.0.1 port 22
    epsilon:notes blyth$  

* in Sharing > Remote Login > add the new user to the list of permitted 

* check can ssh in now, and place the ssh key for passwordless ssh from blyth::

  ssh--putkey F

* minimal setup for using opticks::

    epsilon:~ francis$ ln -s /Users/blyth/opticks
    epsilon:~ francis$ cp ~charles/.bash_profile . 
    epsilon:~ francis$ cp ~charles/.bashrc . 
    epsilon:~ francis$ cp ~charles/.opticks_config . 

    epsilon:~ francis$ cp ~blyth/.vimrc .


* try sharing the rngcache::

    epsilon:~ francis$ mkdir .opticks
    epsilon:~ francis$ cd .opticks
    epsilon:.opticks francis$ ln -s /Users/blyth/.opticks/rngcache
    epsilon:.opticks francis$ 


* customize the config::

    vi ~/.opticks_config   

* build opticks using common opticks source but with different foreign externals::

   ssh F
   opticks-
   opticks-full

* actually its kind of a pity that cannot easily share the opticks "automated" 
  externals unlike the foreign ones which are easily shared 
  

Experimental::

    epsilon:local francis$ ln -s /usr/local/opticks_externals opticks_externals
    epsilon:local francis$ pwd
    /Users/francis/local





console login
--------------

Password:">console"


Time Machine Backup
--------------------


readline : set -o vi
-----------------------

* https://unix.stackexchange.com/questions/30454/advantages-of-using-set-o-vi

* esc v : edit your commandline, then :wq to run it 


Shortcuts
----------

FN key mappings : for failed d D key with F1 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


* http://www.makeuseof.com/tag/remap-macs-function-keys-anything-want/


* SysPrefs > Keyboard > Keyboard, check "Use all F1, F2 etc keys as standard function keys"

  * this disables the special functions of the keys
  * to still use them (change volume/brightness etc)  combine with "fn" 
  * this means pressing those keys enters: <F3><F4>

  * in Terminal > Prefs > Keyboard  then enter that F1 should enter a "d" "D"



invert colors : ctrl-option-cmd-8
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Using functions via sudo su 
------------------------------

::

    delta:opticks blyth$ sudo su 
    sh-3.2# . /Users/blyth/env/osx/osx.bash
    sh-3.2# type osx_adduser
    osx_adduser is a function
    osx_adduser () 
    { 
        local password=${1:-dummy};
        local username=${2:-simon};
        local realname=${3:-Simon};
        local uniqueid=${4:-510};
        local home=/Users/$username;
        [ -d "$home" ] && echo $msg username $username already exists && return;
        [ "$(osx_adduser_check_uniqueid $uniqueid)" != "0" ] && echo $msg uniqueid $uniqueid already used && return;
        dscl . create $home;
        dscl . create $home RealName "$realname";
        dscl . passwd $home $password;
        dscl . create $home UniqueID $uniqueid;
        dscl . create $home PrimaryGroupID 20;
        dscl . create $home NFSHomeDirectory $home;
        dscl . create $home UserShell /bin/bash;
        local group="staff";
        dseditgroup -o edit -t user -a $username $group
    }


osx_simon unable to run CUDA, but fast user switch simon can ?
-----------------------------------------------------------------

::

    delta:build simon$ cudaGetDevicePropertiesTest 
    CUDA Device Query...target -1 
    There are 0 CUDA devices.
    0
    delta:build simon$

::

    delta:build simon$ env
    TERM=xterm-256color
    SHELL=/bin/bash
    USER=simon
    PATH=/Users/blyth/opticks/bin:/usr/local/opticks/lib:/opt/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin:/opt/X11/bin
    PWD=/usr/local/opticks/build
    SHLVL=1
    HOME=/Users/simon
    _=/usr/bin/env
    OLDPWD=/Users/simon



See cuda- : disabling automatic graphics switching, enables cuda usage.


osx_simon : GLFW : No monitors found
--------------------------------------

::

    delta:~ simon$ AxisTest
    2016-07-15 23:38:32.147 INFO  [776196] [main@163] AxisTest
      0 : AxisTest
    2016-07-15 23:38:32.149 INFO  [776196] [Scene::init@199] Scene::init (config from cmake) OGLRAP_INSTALL_PREFIX /usr/local/opticks OGLRAP_SHADER_DIR /usr/local/opticks/gl OGLRAP_SHADER_INCL_PATH /usr/local/opticks/gl OGLRAP_SHADER_DYNAMIC_DIR /usr/local/opticks/gl
    2016-07-15 23:38:32.149 INFO  [776196] [AxisTest::prepareViz@94] AxisTest::prepareViz initRenderers 
    2016-07-15 23:38:32.154 INFO  [776196] [AxisTest::prepareViz@100] AxisTest::prepareViz initRenderers DONE 
    No monitors found



login simon : Cocoa: Failed to retrieve display name
--------------------------------------------------------

Instead of osx_simon using login from terminal window to change user gets further::

    delta:2016 blyth$ login
    login: simon
    Password:
    Last login: Thu Jul 14 22:27:34 on ttys000
    delta:~ simon$ 
    ...
    2016-07-16 09:56:56.819 INFO  [824821] [AxisTest::prepareViz@100] AxisTest::prepareViz initRenderers DONE 
    Cocoa: Failed to retrieve display name
    2016-07-16 09:56:56.988 INFO  [824821] [AxisTest::prepareViz@102] AxisTest::prepareViz frame init DONE 
    2016-07-16 09:56:56.988 INFO  [824821] [AxisTest::prepareViz@105] AxisTest::prepareViz DONE 


::

     43 static char* getDisplayName(CGDirectDisplayID displayID)
     44 {
     45     char* name;
     46     CFDictionaryRef info, names;
     47     CFStringRef value;
     48     CFIndex size;
     49 
     50     // NOTE: This uses a deprecated function because Apple has
     51     //       (as of January 2015) not provided any alternative
     52     info = IODisplayCreateInfoDictionary(CGDisplayIOServicePort(displayID),
     53                                          kIODisplayOnlyPreferredName);
     54     names = CFDictionaryGetValue(info, CFSTR(kDisplayProductName));
     55 
     56     if (!names || !CFDictionaryGetValueIfPresent(names, CFSTR("en_US"),
     57                                                  (const void**) &value))
     58     {
     59         // This may happen if a desktop Mac is running headless
     60         _glfwInputError(GLFW_PLATFORM_ERROR,
     61                         "Cocoa: Failed to retrieve display name");
     62 
     63         CFRelease(info);
     64         return strdup("Unknown");
     65     }



The IODisplayCreateInfoDictionary has been replaced is newer GLFW:

* https://github.com/glfw/glfw/commit/8101d7a7b67fc3414769b25944dc7c02b58d53d0

* http://opensource.apple.com//source/IOKitUser/IOKitUser-388.2/graphics.subproj/IODisplayTest.c



FUNCTIONS
-----------

osx_library_visible
       http://gregferro.com/make-library-folder-visible-in-os-x-lion/
       http://coolestguidesontheplanet.com/show-hidden-library-and-user-library-folder-in-osx/

osx_captive_wifi_disable
      http://apple.stackexchange.com/questions/45418/how-to-automatically-login-to-captive-portals-on-os-x
      https://discussions.apple.com/thread/525840

      The braindead little WebView window that pops up on joining captive portal wifi network does not remember username/password, 
      disabling com.apple.captive.control makes the portal authentication go via Safari which does remember.


osx_ss
      path of last screen shot from today 


osx_ss_copy name



osx_ss_cp name
      copy last screen shot to ~/simoncblyth.bitbucket.org/env/current-relative-dir/name.png
      where current-relative-dir is PWD relative to ENV_HOME

      Thus to use:

         cd ~/env/graphics/ggeoview

         # take screen shot using shift-cmd-4 and dragging a rectangle

         osx_ss-cp name

         # copy-and-paste rst snippet into presentation


::

    simon:pmt blyth$ osx_ss-cp hemi-pmt-parts
    cp "/Users/blyth/Desktop/Screen Shot 2015-10-29 at 11.27.18 AM.png" /Users/blyth/simoncblyth.bitbucket.org/nuwa/detdesc/pmt/hemi-pmt-parts.png
    -rw-r--r--@ 1 blyth  staff  124671 Oct 29 11:29 /Users/blyth/simoncblyth.bitbucket.org/nuwa/detdesc/pmt/hemi-pmt-parts.png

    .. image:: /env/nuwa/detdesc/pmt/hemi-pmt-parts.png
       :width: 900px
       :align: center

    simon:pmt blyth$ pwd
    /Users/blyth/env/nuwa/detdesc/pmt



Upgrade ? Mavericks/Yosemite/El Capitan
------------------------------------------

* :google:`OSX El Capitan NVIDIA GPU`

* http://www.tonymacx86.com/graphics/180741-nvidia-releases-alternate-graphics-drivers-os-x-10-11-2-346-03-04-a.html


* http://www.nvidia.com/object/macosx_cuda-7.5.25-driver.html







EOU
}
osx_dir(){ echo $(local-base)/env/osx/osx_osx ; }
osx_cd(){  cd $(osx_dir); }
osx_get(){
   local dir=$(dirname $(osx_dir)) &&  mkdir -p $dir && cd $dir

}

osx_ss(){
   echo $(ls -1t ~/Desktop/Screen\ Shot\ $(date +'%Y-%m-%d')*.png | head -1 )
}

osx_ss_open(){
   open "$(osx_ss)"
}


osx_mac_address()
{
   networksetup -listallhardwareports 
}


osx_ss_copy(){
   local name=$1
   cp "$(osx_ss)" $name.png

   ipython $(which downsize.py) $name.png
}

osx_ss_copy_invert(){
   local name=$1
   cp "$(osx_ss)" $name.png
   ipython $(which downsize_invert.py) $name.png
}


osx_set_terminal_background_color_notes(){ cat << EON

https://superuser.com/questions/1188772/mac-command-to-change-the-background-color-in-a-terminal

EON
}

osx_set_terminal_background_color()
{ 
   local script=/tmp/$USER/$FUNCNAME.applescript
   mkdir -p $(dirname $script) 

   local r=$1
   local g=$2
   local b=$3
   local w=${4:-1}

   local msg="=== $FUNCNAME : "
   echo $msg w $w r $r g $g b $b

cat << EOS > $script
tell application "Terminal"
   set background color of window $w to {$r, $g, $b}
end tell
EOS

   cat $script 
   osascript $script
}

osx_pink(){   osx_set_terminal_background_color 65535 33667 49601 $1 ; }
osx_white(){  osx_set_terminal_background_color 65535 65535 65535 $1 ; }
osx_grey(){   osx_set_terminal_background_color 30000 30000 30000 $1 ; }





osx_pdf_combo_notes(){ cat << EON

$FUNCNAME
======================

* https://apple.stackexchange.com/questions/230437/how-can-i-combine-multiple-pdfs-using-the-command-line

Have found that this commandline approach succeeds for PDFs that the interactive Preview 
technique (thumbnail dragging or menu) refuses to combine, for unknown reasons. 

1. Usage, create a "combo.sh" listing the pdfs to be combined, and run it::

    #!/bin/bash -l

    osx_
    osx_pdf_combo \
               a.pdf \
               b.pdf \
               c.pdf \
               d.pdf 

2. Do rotations in Preview afterwards.

EON
}

osx_pdf_combo(){
   local cwd=$PWD
   local nam=$(basename $cwd)
   local combo=combo-${nam}.pdf

   "/System/Library/Automator/Combine PDF Pages.action/Contents/Resources/join.py" -o $combo $*  
}




osx_ss_cp(){
   local msg="=== $FUNCNAME :"
   local nam=${1:-plot}
   local iwd=$(realpath ${PWD})
   local src="$(osx_ss)"

   local rel
   local repo

   if [ "${iwd/$ENV_HOME\/}" != ${iwd} ]; then 
       rel=${iwd/$ENV_HOME\/}
       repo="env"

   elif [ "${iwd/$OPTICKS_HOME\/}" != ${iwd} ]; then 
       rel=${iwd/$OPTICKS_HOME\/}
       repo="env"     ## opticks still using env folder in bitbucket statics

   elif [ "${iwd/$WORKFLOW_HOME\/}" != ${iwd} ]; then 
       rel=${iwd/$WORKFLOW_HOME\/}
       repo="workflow"
   else
       echo $msg expects to be run from within env, opticks or workflow repos
       return 
   fi

   local dir
   case $repo in 
            env) dir=$HOME/simoncblyth.bitbucket.io/env/$rel ;;
       workflow) dir=$HOME/DELTA/wdocs/$rel ;;
   esac

   local dst=$dir/$nam.png
   [ ! -d "$dir" ] && mkdir -p $dir

   echo $msg iwd $iwd rel $rel repo $repo dir $dir dst $dst  
  
   ls -l $dir

   local cmd="cp \"$src\" $dst"
   echo $cmd

   if [ -f "$dst" ]; then 
       local ans
       read -p  "Destination file exists already : enter YES to overwrite " ans
       [ "$ans" != "YES" ] && echo skipping && return 
   fi

   eval $cmd
   ls -l $dst

   cat << EOR

.. image:: /$repo/$rel/$nam.png
   :width: 900px
   :align: center

EOR
}





osx_captive_wifi()
{
    type $FUNCNAME
    local arg=${1:-true}
    echo arg $arg
    sudo defaults write /Library/Preferences/SystemConfiguration/com.apple.captive.control Active -boolean $arg
}

osx_captive_wifi_disable(){ osx_captive_wifi false ; }
osx_captive_wifi_enable(){  osx_captive_wifi true ; }




osx_library_visible(){
 
   chflags nohidden ~/Library/
}

osx_prevent_ds_store_droppings_on_shares(){
  defaults write com.apple.desktopservices DSDontWriteNetworkStores true

  cat << EON

Mac OS X v10.4 and later: How to prevent .DS_Store file creation over network connections

https://support.apple.com/en-gb/ht1629

After changing defaults, 
   Either restart the computer or log out and back in to the user account.

If you want to prevent .DS_Store file creation for other users on the same
computer, log in to each user account and perform the steps above—or distribute
a copy of the newly modified com.apple.desktopservices.plist file to the
~/Library/Preferences folder of other user accounts.  These steps do not
prevent the Finder from creating .DS_Store files on the local volume, and these
steps do not prevent previously existing .DS_Store files from being copied to
the remote file server.

Disabling the creation of .DS_Store files on remote file servers can cause
unexpected behavior in the Finder (click here for an example).

https://support.apple.com/kb/TA21373?locale=en_GB

EON

}



osx_adduser_uniqueid_count()
{
   local n=$(dscl . list /Users UniqueID | grep ${1:-501} | wc -l)
   echo $n
}

osx_adduser()
{
    local password=${1:-dummy}
    local username=${2:-simon}
    local realname=${3:-Simon}
    local uniqueid=${4:-510}
    local home=/Users/$username

    [ -d "$home" ] && echo $msg username $username already exists && return 
    [ "$(osx_adduser_uniqueid_count $uniqueid)" != "0" ] && echo $msg uniqueid $uniqueid already used && return 

    dscl . create $home
    dscl . create $home RealName "$realname"
    dscl . passwd $home $password
    dscl . create $home UniqueID $uniqueid
    dscl . create $home PrimaryGroupID 20
    dscl . create $home NFSHomeDirectory $home
    dscl . create $home UserShell /bin/bash
   
    local group="staff"
    dseditgroup -o edit -t user -a $username $group

    createhomedir -c  
}

osx_passwd()
{
    local password=${1:-dummy}
    local username=${2:-simon}
    local home=/Users/$username
    dscl . passwd $home $password
}


osx_simon()
{
    sudo su - simon
}


osx_hide_desktop_icons_notes(){ cat << EON
https://setapp.com/how-to/hide-icons-on-mac

EON
}
osx_desktop_icons_hide()
{
    defaults write com.apple.finder CreateDesktop false
    killall Finder 
}
osx_desktop_icons_hide_not()
{
    defaults write com.apple.finder CreateDesktop true
    killall Finder 
}


osx_pkgutil_notes(){ cat << EON

epsilon:ac3 blyth$ pkgutil --payload-files Anaconda3-2023.07-2-MacOSX-x86_64.pkg | grep seaborn
./anaconda3/pkgs/seaborn-0.12.2-py311hecd8cb5_0
./anaconda3/pkgs/seaborn-0.12.2-py311hecd8cb5_0/info
./anaconda3/pkgs/seaborn-0.12.2-py311hecd8cb5_0/info/repodata_record.json
./anaconda3/pkgs/seaborn-0.12.2-py311hecd8cb5_0.conda
epsilon:ac3 blyth$ 
epsilon:ac3 blyth$ 
epsilon:ac3 blyth$ pwd
/Users/blyth/ac3
epsilon:ac3 blyth$ 

EON
}



osx_sips_notes(){ cat << EON

epsilon:opticks_refs blyth$ sips -g all  Earth.jpg 
/Users/blyth/tree/opticks_refs/Earth.jpg
  pixelWidth: 8192
  pixelHeight: 4096
  typeIdentifier: public.jpeg
  format: jpeg
  formatOptions: default
  dpiWidth: 72.000
  dpiHeight: 72.000
  samplesPerPixel: 3
  bitsPerSample: 8
  hasAlpha: no
  space: RGB
epsilon:opticks_refs blyth$ 

https://nancyisanerd.com/flip-rotate-resize-images-via-command-line-with-sips/

By default, sips rotates clockwise, so you’ll need to specify in degrees if you want to rotate an image:

sips -r 90 image.jpg

https://osxdaily.com/2010/07/13/immediately-resize-rotate-and-flip-images-via-the-command-line/

For anyone wondering how to rotate COUNTER-CLOCKWISE, you have to put the angle
in single quotes. The following code rotates test.image -30 degrees:

sips -r ‘-30’ test.jpg


https://coderwall.com/p/ekhe8g/batch-processing-images-on-mac-with-sips


sips mdls disagreement

https://www.macscripter.net/t/sips-vs-mdls/72332


sips -g pixelHeight -g pixelWidth

mdls -name kMDItemPixelHeight -name kMDItemPixelWidth


Shane_Stanley
May '20

Width and height mean different things in different contexts.

Orientation is a purely metadata concept, designed so photos can automatically be rotated when opened. It makes sense that Spotlight metadata reflects the orientation in its values, given its use for things like Finder info – user-facing values. It’s possible the EXIF result reflects its use in the days before orientation metadata, and is avoiding potential ambiguity.

In the case of sips, the values are used for clipping, scaling, etc, and they therefore need to reflect the pixel values in the actual image data of the file, ignoring any metadata. When sips opens a file, it doesn’t consult the metadata for orientation (or anything else).

So which tool you use should depend on what you’re using the values for.

Here’s an alternative that returns values similar to sips.

use AppleScript version "2.5"
use scripting additions
use framework "Foundation"
use framework "AppKit"

set theFile to posix path of (choose file)
set imageRep to current application's NSBitmapImageRep's imageRepWithContentsOfFile:theFile
set theWidth to imageRep's pixelsWide()
set theHeight to imageRep's pixelsHigh()


* https://www.macscripter.net/t/sips-vs-mdls/72332/20




EON
}

osx_open_notes(){ cat << EON

man open::
    --args
         All remaining arguments are passed to the opened application in the argv parameter to main().  
         These arguments are not opened or interpreted by the open tool.

google:"macOS open --args swift handle argv"


* https://stackoverflow.com/questions/24009050/how-do-i-access-program-arguments-in-swift
* Swift2: Process.arguments
* Swift3: CommandLine.arguments


let args = NSProcessInfo.processInfo().arguments
print(args)


* https://developer.apple.com/documentation/os/logging


EON
}


osx_environment(){ cat << EON

* https://developer.apple.com/documentation/foundation/processinfo/1417911-environment
* https://stackoverflow.com/questions/12165385/how-to-set-environment-variables-to-an-application-on-osx-mountain-lion
* https://stackoverflow.com/questions/603785/environment-variables-in-mac-os-x/4567308#4567308
* https://stackoverflow.com/questions/54724324/access-user-environment-variable-in-swift-for-macos

::

    sudo launchctl setenv EVAR HELLO_WORLD
    sudo launchctl getenv EVAR 
    HELLO_WORLD

Never saw the app getting this one though. 

Starting app from Contents/MacOS with an EVAR succeeds to get the envvar into the 
app : but launcing like this sends no logging to Console::

    epsilon:MacOS blyth$ EVAR=HELLO ./AppName 
    EVAR HELLO

In other launch situations such as when opening .app from Finder
get some logging but envvar values are obfuscated to "<private>"

When launching from Xcode get the values from the 
Product > Scheme menu


EON
}


osx_install_macOS_app_into_applications(){ cat << EON


https://stackoverflow.com/questions/58355389/is-it-possible-to-add-macos-app-from-a-xcode-project-to-launchpad

Yes. Open the Xcode project for the app that you want. Make sure that the
target for your build is "My Mac". Then click Product→Archive. Once a new
window opens with your archive, click Distribute App, then Copy App. Click Next
then choose a location for the app to be put.

Note: you can skip this if you pay for a developer's account (I think). Next,
right-click the app file and click Open. It will say it's from an untrusted
developer (you) because you don't pay for the developer's program.

Now move the app into the Applications folder.



Yes, GO to Product→Archive→Distribute App→Copy App and move to application folder. 


The "Copy App" does : "Export a copy of the archived app"



EON
}

osx_app_path(){ 
   local app=${1:-ImagePreview}
   osascript -e "POSIX path of (path to application \"$app\")"
}

osx_lsregister(){
    /System/Library/Frameworks/CoreServices.framework/Versions/A/Frameworks/LaunchServices.framework/Versions/A/Support/lsregister $* 
}

osx_app_find(){
   local app=${1:-ImagePreview}
   osx_lsregister -dump | grep -o "/.*${app}.app" | grep -v -E "Caches|TimeMachine|Temporary|/Volumes/$app" | uniq
}


