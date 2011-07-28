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


EOU
}
videola-dir(){ echo $(local-base)/env/videola ; }
videola-cd(){  cd $(videola-dir); }
videola-mate(){ mate $(videola-dir) ; }
videola-get(){
   local dir=$(dirname $(videola-dir)) &&  mkdir -p $dir && cd $dir

   git clone git://github.com/Lullabot/videola.git
}
