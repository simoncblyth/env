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

   man launchctl



Listing All Daemons
-----------------------------------

NB need to sudo to list em all

::

    delta:LaunchDaemons blyth$ launchctl list | wc -l
         268
    delta:LaunchDaemons blyth$ sudo launchctl list | wc -l
         585


::

    delta:LaunchDaemons blyth$ launchctl list|grep apache

    delta:LaunchDaemons blyth$ sudo launchctl list|grep apache
    62  -   org.apache.httpd


Listing Info for one Daemon
-----------------------------

::

    delta:LaunchDaemons blyth$ sudo launchctl list org.apache.httpd
    {
        "Label" = "org.apache.httpd";
        "LimitLoadToSessionType" = "System";
        "OnDemand" = false;
        "LastExitStatus" = 0;
        "PID" = 62;
        "TimeOut" = 30;
        "ProgramArguments" = (
            "/usr/sbin/httpd";
            "-D";
            "FOREGROUND";
        );
    };


    delta:LaunchDaemons blyth$ sudo launchctl list -x org.apache.httpd 
    <?xml version="1.0" encoding="UTF-8"?>
    <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
    <plist version="1.0">
    <dict>
        <key>Label</key>
        <string>org.apache.httpd</string>
        <key>LastExitStatus</key>
        <integer>0</integer>
        <key>LimitLoadToSessionType</key>
        <string>System</string>
        <key>OnDemand</key>
        <false/>
        <key>PID</key>
        <integer>62</integer>
        <key>ProgramArguments</key>
        <array>
            <string>/usr/sbin/httpd</string>
            <string>-D</string>
            <string>FOREGROUND</string>
        </array>
        <key>TimeOut</key>
        <integer>30</integer>
    </dict>
    </plist>




   delta:arc blyth$ cat /System/Library/LaunchDaemons/org.apache.httpd.plist
    <?xml version="1.0" encoding="UTF-8"?>
    <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
    <plist version="1.0">
    <dict>
        <key>Disabled</key>
        <true/>
        <key>Label</key>
        <string>org.apache.httpd</string>
        <key>EnvironmentVariables</key>
        <dict>
            <key>XPC_SERVICES_UNAVAILABLE</key>
            <string>1</string>
        </dict>
        <key>ProgramArguments</key>
        <array>
            <string>/usr/sbin/httpd</string>
            <string>-D</string>
            <string>FOREGROUND</string>
        </array>
        <key>OnDemand</key>
        <false/>
    </dict>
    </plist>
    delta:arc blyth$ 



Non System Daemons
---------------------

::

    delta:arc blyth$ l /Library/LaunchDaemons/
    total 56
    -rw-r--r--  1 root  wheel  664 Oct 26  2016 org.macosforge.xquartz.privileged_startx.plist
    lrwxr-xr-x  1 root  admin   66 Oct 17  2016 org.freedesktop.dbus-system.plist -> /opt/local/Library/LaunchDaemons/org.freedesktop.dbus-system.plist
    lrwxr-xr-x  1 root  wheel  103 Jul 23  2015 com.oracle.java.Helper-Tool.plist -> /Library/Internet Plug-Ins/JavaAppletPlugin.plugin/Contents/Resources/com.oracle.java.Helper-Tool.plist
    -rw-r--r--  1 root  wheel  485 Jun 29  2015 com.nvidia.cuda.launcher.plist
    -rw-r--r--  1 root  wheel  656 Mar 23  2015 com.daz3d.content_management_service.plist
    lrwxr-xr-x  1 root  admin   90 Feb 12  2015 org.macports.mysql56-server.plist -> /opt/local/etc/LaunchDaemons/org.macports.mysql56-server/org.macports.mysql56-server.plist
    lrwxr-xr-x  1 root  admin   72 Nov 23  2013 org.macports.nginx.plist -> /opt/local/etc/LaunchDaemons/org.macports.nginx/org.macports.nginx.plist




Misc
-------

::

    b2mc:00101 heprez$ sudo launchctl getenv PATH
    /usr/bin:/bin:/usr/sbin:/sbin
    b2mc:00101 heprez$ which java
    /usr/bin/java


::
   
   sudo launchctl load -w /Library/LaunchDaemons/org.heprez.exist.plist  
   # appears need the "-w" whrn Disable us true otherwise get "nothing to load" message


::

    delta:com.apple.launchd blyth$ launchctl list org.openbsd.ssh-agent
    {
        "Label" = "org.openbsd.ssh-agent";
        "LimitLoadToSessionType" = "Aqua";
        "OnDemand" = true;
        "LastExitStatus" = 0;
        "TimeOut" = 30;
        "ProgramArguments" = (
            "/usr/bin/ssh-agent";
            "-l";
        );
        "EnableTransactions" = true;
        "Sockets" = {
            "Listeners" = (
                file-descriptor-object;
            );
        };
    };
    delta:com.apple.launchd blyth$ pwd
    /var/log/com.apple.launchd




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




