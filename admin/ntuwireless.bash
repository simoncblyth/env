# === func-gen- : admin/ntuwireless fgp admin/ntuwireless.bash fgn ntuwireless fgh admin
ntuwireless-src(){      echo admin/ntuwireless.bash ; }
ntuwireless-source(){   echo ${BASH_SOURCE:-$(env-home)/$(ntuwireless-src)} ; }
ntuwireless-vi(){       vi $(ntuwireless-source) ; }
ntuwireless-env(){      elocal- ; }
ntuwireless-usage(){ cat << EOU

NTU Wireless 
==============

Issue : slow initial connection 
---------------------------------

Attempting to gain connectivity to ntuwireless at the
start of the day it is taking 2~3 minutes for Safari 
to get to the login form. 

Suspect the problem was with the provider, 
as seems to have resolved itself with no change on my part.

But curl gets to the form immediately.::

   curl -L -v http://www.google.com > ~/ntuwireless.txt 2>&1

Perhaps can find way to auto login using curl.


iPod Touch Fails to remember the username/password NTU "captive wifi" 
---------------------------------------------------------------------------------

* https://support.apple.com/en-us/HT204497

Captive Wi-Fi networks are public Wi-Fi networks that you subscribe to or pay
to use. These networks are also called "subscription" or "Wi-Fi Hotspot"
networks.


#. in network settings for "NTU", *Auto-Join* was OFF and *Auto-login* was ON 
#. try setting those both ON





EOU
}
ntuwireless-dir(){ echo $(local-base)/env/admin/admin-ntuwireless ; }
ntuwireless-cd(){  cd $(ntuwireless-dir); }
ntuwireless-mate(){ mate $(ntuwireless-dir) ; }
ntuwireless-get(){
   local dir=$(dirname $(ntuwireless-dir)) &&  mkdir -p $dir && cd $dir

}
