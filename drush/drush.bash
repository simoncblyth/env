# === func-gen- : drush/drush fgp drush/drush.bash fgn drush fgh drush
drush-src(){      echo drush/drush.bash ; }
drush-source(){   echo ${BASH_SOURCE:-$(env-home)/$(drush-src)} ; }
drush-vi(){       vi $(drush-source) ; }
drush-env(){      
   elocal- 
   env-append $(drush-dir)
}
drush-usage(){
  cat << EOU
     drush-src : $(drush-src)
     drush-dir : $(drush-dir)

     drush : drupal shell (minimum PHP version is 5.2 so nogo on C,C2,N,H,... restricted to G)
        http://drupal.org/project/drush

     `drush-get-make` installs the drush_make extension to drush into ~/.drush
        http://drupal.org/project/drush_make

      after running `drush help` lists the additional command 




EOU
}
drush-dir(){ echo $(local-base)/env/drush/drush ; }
drush-cd(){  cd $(drush-dir); }
drush-mate(){ mate $(drush-dir) ; }
drush-name(){ echo drush-7.x-4.4 ; }
drush-make-name(){ echo drush_make-6.x-2.2 ; }

drush-get(){
   local dir=$(dirname $(drush-dir)) &&  mkdir -p $dir && cd $dir

   local tgz=$(drush-name).tar.gz
   [ ! -f "$tgz" ]          && curl -L -O http://ftp.drupal.org/files/projects/$tgz
   [ ! -d "drush" ]         && tar zxvf $tgz

}

drush-get-make(){
   mkdir -p ~/.drush
   cd ~/.drush
   local tgz=$(drush-make-name).tar.gz 
   [ ! -f "$tgz" ]         && curl -L -O http://ftp.drupal.org/files/projects/$tgz
   [ ! -d "drush_make" ]    && tar zxvf $tgz 
}




