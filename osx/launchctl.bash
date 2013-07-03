# === func-gen- : osx/launchctl fgp osx/launchctl.bash fgn launchctl fgh osx
launchctl-src(){      echo osx/launchctl.bash ; }
launchctl-source(){   echo ${BASH_SOURCE:-$(env-home)/$(launchctl-src)} ; }
launchctl-vi(){       vi $(launchctl-source) ; }
launchctl-env(){      elocal- ; }
launchctl-usage(){ cat << EOU

LAUNCHCTL
==========

* :google:`launchctl`
* http://developer.apple.com/library/mac/#documentation/Darwin/Reference/ManPages/man1/launchctl.1.html
* http://developer.apple.com/library/mac/#documentation/Darwin/Reference/ManPages/man5/launchd.plist.5.html#//apple_ref/doc/man/5/launchd.plist

::

    b2mc:00101 heprez$ sudo launchctl getenv PATH
    /usr/bin:/bin:/usr/sbin:/sbin
    b2mc:00101 heprez$ which java
    /usr/bin/java


::
   
   sudo launchctl load -w /Library/LaunchDaemons/org.heprez.exist.plist  
   # appears need the "-w" whrn Disable us true otherwise get "nothing to load" message


EOU
}
launchctl-dir(){ echo $(local-base)/env/osx/osx-launchctl ; }
launchctl-cd(){  cd $(launchctl-dir); }
launchctl-mate(){ mate $(launchctl-dir) ; }

launchctl-cfg-name(){ echo org.env.demo ; }
launchctl-cfg-path(){ echo /Library/LaunchDaemons/$(launchctl-cfg-name).plist ; }
launchctl-cfg(){
   local msg=" == $FUNCNAME "
   local tgt=$(launchctl-cfg-path)
   local tmp=/tmp/$FUNCNAME/$(basename $tgt) && mkdir -p $(dirname $tmp)

   $FUNCNAME- > $tmp

   cat $tmp 
   local ans 
   read -p "$msg write above filled template $tmp to target $tgt ? YES to proceed: " ans 
   if [ "$ans" == "YES" ]; then  
       local cmd
       cmd="sudo cp $tmp $tgt"
       echo $msg $cmd
       eval $cmd
       cmd="sudo chmod go-rwx $tgt"
       echo $msg $cmd
       eval $cmd
   else
       echo $msg skipping
   fi  
   rm $tmp
   echo $msg must \"launchctl-load\" to start demo daemon
}

launchctl-load(){
   local cmd=$(launchctl-launchctl load)
   echo $cmd
   eval $cmd
}
launchctl-unload(){
   local cmd=$(launchctl-launchctl unload)
   echo $cmd
   eval $cmd
}
launchctl-launchctl(){
   echo sudo launchctl $1 $(launchctl-cfg-path)
}

launchctl-script(){ echo $ENV_HOME/osx/demo_launchctl.sh ; }
launchctl-stdout(){ echo /tmp/demo_launchctl.out ; }
launchctl-stderr(){ echo /tmp/demo_launchctl.err ; }

launchctl-cfg-(){ cat << EOP
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple Computer//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Disabled</key>
    <false/>
    <key>RunAtLoad</key>
    <true/>
    <key>OnDemand</key>
    <false/>
    <key>UserName</key>
    <string>$(id -nu)</string>
    <key>GroupName</key>
    <string>$(id -ng)</string>
    <key>Label</key>
    <string>$(launchctl-cfg-name)</string>
    <key>EnvironmentVariables</key>
    <dict>
           <key>ENV_HOME</key>
           <string>$(env-home)</string>
    </dict>
    <key>ProgramArguments</key>
    <array>
        <string>$(launchctl-script)</string>
    </array>
    <key>StandardOutPath</key>
    <string>$(launchctl-stdout)</string>
    <key>StandardErrorPath</key>
    <string>$(launchctl-stderr)</string>
    <key>LaunchOnlyOnce</key>
    <true/>
</dict>
</plist>
EOP
}




