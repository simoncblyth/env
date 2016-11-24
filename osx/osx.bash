# === func-gen- : osx/osx fgp osx/osx.bash fgn osx fgh osx
osx_src(){      echo osx/osx.bash ; }
osx_source(){   echo ${BASH_SOURCE:-$ENV_HOME/$(osx_src)} ; }
osx_vi(){       vi $(osx_source) ; }
osx_env(){      elocal- ; }
osx_usage(){ cat << EOU


console login
--------------

Password:">console"


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
   downsize.py $name.png
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
   elif [ "${iwd/$WORKFLOW_HOME\/}" != ${iwd} ]; then 
       rel=${iwd/$WORKFLOW_HOME\/}
       repo="workflow"
   else
       echo $msg expects to be run from within env or workfloat repos
       return 
   fi

   local dir
   case $repo in 
            env) dir=$HOME/simoncblyth.bitbucket.org/env/$rel ;;
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
