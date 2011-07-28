# === func-gen- : videola fgp ./videola.bash fgn videola fgh .
videola-src(){      echo videola/videola.bash ; }
videola-source(){   echo ${BASH_SOURCE:-$(env-home)/$(videola-src)} ; }
videola-vi(){       vi $(videola-source) ; }
videola-env(){      elocal- ; }
videola-usage(){
  cat << EOU
     videola-src : $(videola-src)
     videola-dir : $(videola-dir)

     http://www.lullabot.com/articles/introducing-videola
     http://www.videola.tv/
     http://drupalize.me/

     http://drush.ws/
     http://drupalcode.org/project/drush.git/blob/HEAD:/README.txt

           requires at least  PHP 5.2

      yum info php
          N 5.1.6 

    maybe more convenient kickstart method...
          drush make https://raw.github.com/Lullabot/videola/master/videola_starter.make videola


   == alternates ==

      MediaMosa is REST/drupal7 based video backend with transcoding etc...
          http://www.mediamosa.org/

   == installation ==

      videola-get
      videola-init   ## say y 

     cd /Library/WebServer/Documents    
     ln -s /tmp/vdo vdo
     visit    http://localhost/vdo

     warning of no GD
       http://www.gigoblog.com/2008/10/08/install-gd-for-php-on-mac-os-x-105-leopard/

    unfortunately using /usr/bin/php ... 
    maybe start again with macports php for easy gd instalation 
       http://2tbsp.com/content/install_apache_2_and_php_5_macports

EOU
}
videola-dir(){ echo $(local-base)/env/videola ; }
videola-rootdir(){ echo /tmp/vdo ; }
videola-cd(){  cd $(videola-dir); }
videola-rcd(){  cd $(videola-rootdir); }
videola-mate(){ mate $(videola-dir) ; }
videola-get(){
   local dir=$(dirname $(videola-dir)) &&  mkdir -p $dir && cd $dir

   git clone git://github.com/Lullabot/videola.git
}

 

videola-init(){

  local msg="=== $FUNCNAME :"
  local drd=$(videola-rootdir)
  mkdir -p $drd
  cd $drd
  cp $(videola-dir)/videola.make .
  cp $(videola-dir)/videola_starter.make .

  echo $msg from $PWD
  drush make videola_starter.make

}

videola-settings(){

   videola-rcd

   cp ./sites/default/default.settings.php ./sites/default/settings.php
   chmod ugo+rwx ./sites/default/settings.php

   mkdir -p ./sites/default/files 
   chmod ugo+rwx ./sites/default/files


}






