
phpbb-src(){     echo phpbb/phpbb.bash ; }
phpbb-source(){  echo $(env-home)/$(phpbb-src) ; }
phpbb-vi(){      vi $(phpbb-source) ; }
phpbb-env(){     elocal- ; }

phpbb-usage(){
  cat << EOU

     phpbb-name : $(phpbb-name)
     phpbb-zip  : $(phpbb-zip)
     phpbb-url  : $(phpbb-url)
     phpbb-pkgd : $(phpbb-pkgd) 


     phpbb-hookup :
         modify the apache config, uncommenting the php5 LoadModule line
         and planting a link from `apache-htdocs` to the unzip folder `phpbb-dir`
         will need a restart in addition 
              sudo apachectl restart

     phpbb-sqlite
         setup an sqlite database for use by phpbb



     phpbb-install
          web based install
          for sqlite enter path to a db eg  /tmp/demo.db
         
          it wants to edit a config.php


     http://www.phpbb.com/community/viewtopic.php?f=46&t=807865&start=0&st=0&sk=t&sd=a



EOU

}


phpbb-name(){    echo phpBB-3.0.4 ; }
phpbb-pkgd(){    echo phpBB3 ; }
phpbb-urlb(){    echo phpBB3 ; }
   
phpbb-dir(){     echo /tmp ; }
phpbb-zip(){     echo $(phpbb-name).zip; }
phpbb-urlz(){    echo http://d10xg45o6p6dbl.cloudfront.net/projects/p/phpbb/$(phpbb-zip) ; }
phpbb-get(){
  cd $(phpbb-dir)
  [ ! -f $(phpbb-zip) ]  && curl -o $(phpbb-zip) $(phpbb-urlz)  
  [ ! -d $(phpbb-pkgd) ] && unzip $(phpbb-zip) 

  $SUDO chown -R $(apache-user):$(apache-group) $(phpbb-pkgd)  

   ## in particular the config.php needs to be writable
}

phpbb-reget(){
  cd $(phpbb-dir)
  local cmd="rm -rf $(phpbb-pkgd) "
  $SUDO $cmd
  phpbb-get
  phpbb-install
}

phpbb-hookup(){
  cd $(phpbb-dir)
  phpbb-ln
  phpbb-apacheconf
}

phpbb-install(){ open $(phpbb-url install/index.php) ; }

phpbb-ln(){
   local iwd=$PWD
   apache-
   cd `apache-htdocs`
   ln -s $iwd/$(phpbb-pkgd) $(phpbb-urlb)
   cd $iwd
}

phpbb-url(){ echo http://localhost/$(phpbb-urlb)/$1 ; }

phpbb-stamp(){ date +"%Y%m%d-%H%M" ; }

phpbb-apacheconf(){
   local msg="=== $FUNCNAME :"
   apache-
   local tmp=/tmp/env/$FUNCNAME/$(basename $(apache-conf)) && mkdir -p $(dirname $tmp) 
   local cnf=$(apache-conf)
   cp $cnf $tmp
   perl -pi -e 's,#(LoadModule php5_module.*),$1,' $tmp

   echo $msg cnf $cnf 
   diff $cnf $tmp
   $SUDO cp $cnf $cnf.$(phpbb-stamp) 
   $SUDO cp $tmp $cnf

}


phpbb-dbpath(){ echo /private/tmp/demo.db ; }

phpbb-db-create(){
  ## this should live in sqlite.bash 
   local dbp=$1
   #rm $dbp
   sqlite3 $dbp << EOC
      create table t1(one varchar(10));
EOC
}

phpbb-sqlite(){
   cd $(phpbb-dir)
   local dbp=$(phpbb-dbpath)
   echo $msg creating $dbp
   phpbb-db-create $dbp
   apache-
   $SUDO chown $(apache-user):$(apache-group) $dbp

}



