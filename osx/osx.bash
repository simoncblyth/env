# === func-gen- : osx/osx fgp osx/osx.bash fgn osx fgh osx
osx-src(){      echo osx/osx.bash ; }
osx-source(){   echo ${BASH_SOURCE:-$(env-home)/$(osx-src)} ; }
osx-vi(){       vi $(osx-source) ; }
osx-env(){      elocal- ; }
osx-usage(){ cat << EOU




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



EOU
}
osx-dir(){ echo $(local-base)/env/osx/osx-osx ; }
osx-cd(){  cd $(osx-dir); }
osx-mate(){ mate $(osx-dir) ; }
osx-get(){
   local dir=$(dirname $(osx-dir)) &&  mkdir -p $dir && cd $dir

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


