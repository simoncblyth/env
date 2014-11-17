# === func-gen- : messaging/jsmn fgp messaging/jsmn.bash fgn jsmn fgh messaging
jsmn-src(){      echo messaging/jsmn.bash ; }
jsmn-source(){   echo ${BASH_SOURCE:-$(env-home)/$(jsmn-src)} ; }
jsmn-vi(){       vi $(jsmn-source) ; }
jsmn-env(){      elocal- ; }
jsmn-usage(){ cat << EOU

JSMN : pronounced Jasmine
==========================

Extremly minimal JSON parser, it just tokenizes.

* http://zserge.bitbucket.org/jsmn.html

* http://alisdair.mcdiarmid.org/2012/08/14/jsmn-example.html



Other C/C++ JSON parsers
-------------------------

* gason- disqualified by use of C++11


From JSON to SQLite
--------------------

* https://github.com/alinz/JSON-2-SQLite/blob/master/JSON2SQLite.cpp





EOU
}
jsmn-dir(){ echo $(local-base)/env/messaging/jsmn ; }
jsmn-sdir(){ echo $(env-home)/messaging/jsmn ; }
jsmn-cd(){  cd $(jsmn-dir); }
jsmn-scd(){  cd $(jsmn-sdir); }
jsmn-mate(){ mate $(jsmn-dir) ; }
jsmn-get(){
   local dir=$(dirname $(jsmn-dir)) &&  mkdir -p $dir && cd $dir

   hg clone http://bitbucket.org/zserge/jsmn jsmn

}
