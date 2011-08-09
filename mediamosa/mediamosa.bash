# === func-gen- : mediamosa/mediamosa fgp mediamosa/mediamosa.bash fgn mediamosa fgh mediamosa
mediamosa-src(){      echo mediamosa/mediamosa.bash ; }
mediamosa-source(){   echo ${BASH_SOURCE:-$(env-home)/$(mediamosa-src)} ; }
mediamosa-vi(){       vi $(mediamosa-source) ; }
mediamosa-env(){      elocal- ; }
mediamosa-usage(){
  cat << EOU
     mediamosa-src : $(mediamosa-src)
     mediamosa-dir : $(mediamosa-dir)

     http://mediamosa.org/
     http://www.mediamosa.org/trac/wiki


MediaMosa is a full featured, webservice oriented media management and distribution platform. 
MediaMosa is implemented as a modular extension to the open source Drupal system. 
You'll find the MediaMosa specific modules in sites/all/modules, the remainder of the tree is the standard Drupal core.

Before proceeding, please visit our quick_install.

== Install ==

   #. mediamosa-get
   #. mediamosa-apache
   #. restart apache 
   #. point browser at http://localhost/mediamosa
         blurb directs to http://mediamosa.org/trac/wiki/Quick%20install    


== drupal and nginx ==

   http://wiki.nginx.org/Drupal
         looks like good config starting point 
   http://wiki.nginx.org/Pitfalls
         informative ... improving nginx config
   
 
== preqs ==

  http://mediamosa.org/trac/wiki/Pre%20requirement

Apache 2.2.x webserver (or nginx)
   Enable rewrite, mime_magic modules.
   PHP module SAPI/FCGI.
PHP 5.2.x
   Enable cli, curl, gd, mcrypt, mysql modules.
   Optional: install PEAR and PHP development files for APC support.
MySQL 5.x database server (or MariaDB)
   Recommended: Set default tablespace type to InnoDB.
Lua 5
ffmpeg video encoder tool




EOU
}
mediamosa-cd(){  cd $(mediamosa-dir); }
mediamosa-dir(){ echo $(local-base)/env/mediamosa/$(mediamosa-name) ; }
mediamosa-mate(){ mate $(mediamosa-dir) ; }

mediamosa-name(){ echo mediamosa-2.3.13 ; }
mediamosa-ext(){  echo _1 ; }
mediamosa-tgz(){  echo $(mediamosa-name)$(mediamosa-ext).tgz ; }
mediamosa-url(){ echo http://mediamosa.org/sites/default/files/$(mediamosa-tgz) ; }

mediamosa-get(){
   local dir=$(dirname $(mediamosa-dir)) &&  mkdir -p $dir && cd $dir
   local tgz=$(mediamosa-tgz)
   [ ! -f "$tgz" ] && curl -L -O $(mediamosa-url)
   [ ! -d "$(mediamosa-name)" ] && tar zxvf $tgz

}

mediamosa-apache(){

 apache- 
 apache-ln $(mediamosa-dir) mediamosa

}


