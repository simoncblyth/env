# === func-gen- : osx/osx fgp osx/osx.bash fgn osx fgh osx
osx-src(){      echo osx/osx.bash ; }
osx-source(){   echo ${BASH_SOURCE:-$(env-home)/$(osx-src)} ; }
osx-vi(){       vi $(osx-source) ; }
osx-env(){      elocal- ; }
osx-usage(){ cat << EOU


console login
--------------

Password:">console"





FUNCTIONS
-----------

osx-library-visible
       http://gregferro.com/make-library-folder-visible-in-os-x-lion/
       http://coolestguidesontheplanet.com/show-hidden-library-and-user-library-folder-in-osx/

osx-captive-wifi-disable
      http://apple.stackexchange.com/questions/45418/how-to-automatically-login-to-captive-portals-on-os-x
      https://discussions.apple.com/thread/525840

      The braindead little WebView window that pops up on joining captive portal wifi network does not remember username/password, 
      disabling com.apple.captive.control makes the portal authentication go via Safari which does remember.


osx-ss
      path of last screen shot from today 

osx-ss-cp name
      copy last screen shot to ~/simoncblyth.bitbucket.org/env/current-relative-dir/name.png
      where current-relative-dir is PWD relative to ENV_HOME

      Thus to use:

         cd ~/env/graphics/ggeoview

         # take screen shot using shift-cmd-4 and dragging a rectangle

         osx-ss-cp name

         # copy-and-paste rst snippet into presentation


::

    simon:pmt blyth$ osx-ss-cp hemi-pmt-parts
    cp "/Users/blyth/Desktop/Screen Shot 2015-10-29 at 11.27.18 AM.png" /Users/blyth/simoncblyth.bitbucket.org/nuwa/detdesc/pmt/hemi-pmt-parts.png
    -rw-r--r--@ 1 blyth  staff  124671 Oct 29 11:29 /Users/blyth/simoncblyth.bitbucket.org/nuwa/detdesc/pmt/hemi-pmt-parts.png

    .. image:: /env/nuwa/detdesc/pmt/hemi-pmt-parts.png
       :width: 900px
       :align: center

    simon:pmt blyth$ pwd
    /Users/blyth/env/nuwa/detdesc/pmt




EOU
}
osx-dir(){ echo $(local-base)/env/osx/osx-osx ; }
osx-cd(){  cd $(osx-dir); }
osx-mate(){ mate $(osx-dir) ; }
osx-get(){
   local dir=$(dirname $(osx-dir)) &&  mkdir -p $dir && cd $dir

}

osx-ss(){
   echo $(ls -1t ~/Desktop/Screen\ Shot\ $(date +'%Y-%m-%d')*.png | head -1 )
}

osx-ss-open(){
   open "$(osx-ss)"
}


osx-ss-cp(){
   local msg="=== $FUNCNAME :"
   local nam=${1:-plot}
   local iwd=$(realpath ${PWD})
   local src="$(osx-ss)"

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





osx-captive-wifi()
{
    type $FUNCNAME
    local arg=${1:-true}
    echo arg $arg
    sudo defaults write /Library/Preferences/SystemConfiguration/com.apple.captive.control Active -boolean $arg
}

osx-captive-wifi-disable(){ osx-captive-wifi false ; }
osx-captive-wifi-enable(){  osx-captive-wifi true ; }




osx-library-visible(){
 
   chflags nohidden ~/Library/
}

osx-prevent-ds-store-droppings-on-shares(){
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


