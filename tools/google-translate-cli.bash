# === func-gen- : tools/google-translate-cli fgp tools/google-translate-cli.bash fgn google-translate-cli fgh tools
google-translate-cli-src(){      echo tools/google-translate-cli.bash ; }
google-translate-cli-source(){   echo ${BASH_SOURCE:-$(env-home)/$(google-translate-cli-src)} ; }
google-translate-cli-vi(){       vi $(google-translate-cli-source) ; }
google-translate-cli-env(){      elocal- ; }
google-translate-cli-usage(){ cat << EOU

GOOGLE TRANSLATE COMMANDLINE
=============================

Paste in translation, I hope.

* https://github.com/soimort/google-translate-cli

OSX
----

::

    sudo port install -v gawk 


::

    simon:Desktop blyth$ echo bonjour monde | trs -
    hello world


EOU
}
google-translate-cli-dir(){ echo $(local-base)/env/tools/tools-google-translate-cli ; }
google-translate-cli-cd(){  cd $(google-translate-cli-dir); }
google-translate-cli-mate(){ mate $(google-translate-cli-dir) ; }
google-translate-cli-get(){
   local dir=$(dirname $(google-translate-cli-dir)) &&  mkdir -p $dir && cd $dir

   [ ! -d google-translate-cli ] && git clone git://github.com/soimort/google-translate-cli.git   

   [ -f /usr/bin/translate ] && echo already installed && return 1 
   [ -f /usr/bin/trs ]      && echo already installed && return 1 

   cd google-translate-cli
   make install  

}
