console-vi(){ vi $BASH_SOURCE ; }
console-env(){ echo -n ; }
console-usage(){ cat << EOU
Console.app
=============

swift logging : prior to swift 4/5? Logger struct
---------------------------------------------------

::

    import os.log 
    let customLog = OSLog(subsystem: "com.simon", category: "launch")
    os_log("An error occurred!", log: customLog, type: .error)


man log 
--------

::

   log show --info --debug --predicate 'eventType == logEvent and subsystem contains "com.simon" '

   sudo log show --info --debug --predicate 'eventType == logEvent and subsystem contains "com.simon" ' 
   sudo log show --info --debug --predicate 'subsystem == "com.simon" ' 

This is more live::

   sudo log stream --info --debug --predicate 'subsystem contains "com.simon" '  


Making a symbolic link to the app and using Finder.app to launch does write to the log.

HMM: DOES SHOW UP BUT WITH DELAY, AND NOT WHEN LAUNCHING WITH OPEN::


    epsilon:osx blyth$ sudo log show --info --debug --predicate 'eventType == logEvent and subsystem contains "com.simon" '
    Password:
    Filtering the log data using "eventType == 1024 AND subsystem CONTAINS "com.simon""
    Timestamp                       Thread     Type        Activity             PID    TTL  
    2023-10-14 17:53:54.688700+0800 0x1e2cd    Error       0x0                  4004   14   ImagePreview: [com.simon:launch] An error occurred!
    2023-10-14 17:54:50.207354+0800 0x1e584    Error       0x0                  4009   14   ImagePreview: [com.simon:launch] An error occurred!
    2023-10-14 17:57:06.562904+0800 0x1e9a2    Error       0x0                  4019   14   ImagePreview: [com.simon:launch] An error occurred!
    2023-10-14 17:57:18.964840+0800 0x1ea44    Error       0x0                  4021   14   ImagePreview: [com.simon:launch] An error occurred!
     --------------------------------------------------------------------------------------------------------------------
    Log      - Default:          0, Info:                0, Debug:             0, Error:          4, Fault:          0
    Activity - Create:           0, Transition:  


::

    epsilon:gallery blyth$ sudo log config --subsystem com.simon
    Mode for 'com.simon'  INFO PERSIST_DEFAULT
    epsilon:gallery blyth$ 




Some logging bug ?
-------------------

* https://developer.apple.com/forums/thread/82736
 

finding log messages : not easy as so many of them
----------------------------------------------------

* https://support.apple.com/guide/console/log-messages-cnsl1012/mac

In the Console app on your Mac, in the Devices list on the left, select the
device you want to view log messages for (such as your Mac, iPhone, iPad, Apple
Watch, or Apple TV). If you don’t see the Devices list, click the Sidebar
button in the Favorites bar.

* https://support.omnigroup.com/console-osx/


issue : can find messages when launcing with xcode but not with open 
----------------------------------------------------------------------

Also getting obscured "<private>" values for envvars, see osx_environment_notes 


Recording Private Data in the System Log, un-obfuscating those "<private>"
---------------------------------------------------------------------------

* https://developer.apple.com/forums/thread/705810

There are two ways to enable recording of private data in the system log:

* Add an OSLogPreferences property in your app profile (all platforms).
* Create and install a custom configuration profile with the SystemLogging (com.apple.system.logging) payload (macOS only).







EOU
}


console-stream(){
   type $FUNCNAME 
   sudo log stream --info --debug --predicate 'subsystem contains "com.simon" ' 
}


