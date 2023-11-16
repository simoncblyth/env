firefox-env(){ echo -n ;}
firefox-vi(){ vi $BASH_SOURCE ; }
firefox-usage(){ cat << EOU
tools/firefox.bash 
=====================


Issue : All of a sudden "Firefox Developer Edition.app" not starting from macOS Dock it just bounces
------------------------------------------------------------------------------------------------------

Check in Console.app but not informative. Found the "Force quit" kill to: org.mozilla.firefoxdeveloperedition 

Also fails to open with::

   open -a "Firefox Developer Edition.app" 

BUT: it does start from the commandline::

   cd /Applications/Firefox\ Developer\ Edition.app/Contents/MacOS/
   ./firefox 

Despite the repeated warning it seems to work OK::

    epsilon:MacOS blyth$ ./firefox
    _RegisterApplication(), FAILED TO establish the default connection to the WindowServer, _CGSDefaultConnection() is NULL.
    _RegisterApplication(), FAILED TO establish the default connection to the WindowServer, _CGSDefaultConnection() is NULL.
    ...

The suddenly failing makes me think of expired cerificates. 


Search : "macOS Firefox Developer Edition cannot launch from Dock"
--------------------------------------------------------------------

* https://support.mozilla.org/en-US/questions/989916

macOS : Open in safe mode by pressing option key when opening   


EOU
}


firefox-cd(){
  cd /Applications/Firefox\ Developer\ Edition.app/Contents/MacOS
  pwd

}


firefox-open(){
  firefox-cd

  ./firefox

}

