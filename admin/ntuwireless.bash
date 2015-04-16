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

But curl gets to the form immediately.::

   curl -L -v http://www.google.com > ~/ntuwireless.txt 2>&1

Perhaps can find way to auto login using curl.



EOU
}
ntuwireless-dir(){ echo $(local-base)/env/admin/admin-ntuwireless ; }
ntuwireless-cd(){  cd $(ntuwireless-dir); }
ntuwireless-mate(){ mate $(ntuwireless-dir) ; }
ntuwireless-get(){
   local dir=$(dirname $(ntuwireless-dir)) &&  mkdir -p $dir && cd $dir

}
