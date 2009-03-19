
phpbb-dbg(){     bash $(phpbb-source) ; }
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
         
          `apache-user` needs write access to config.php

           was getting error to connect to sqlite db .. 
                    http://www.phpbb.com/community/viewtopic.php?f=46&t=807865&start=0&st=0&sk=t&sd=a
           afted adjusting to assign db into a controlled dir owned by apache-user and
           ensure that it doesnt exist to begin with succeed to connect 

                -rw-r--r--  1 _www  _www   0 19 Mar 15:44 demo.db
                -rw-------  1 _www  _www  20 19 Mar 15:45 demo.db-journal

             $dbhost = '/private/tmp/phpbb-dbpath/demo.db';
             $dbname = 'demo';

            BUT
              get a blank screen at "create_table" stage : 
                    http://localhost/phpBB3/install/index.php?mode=install&sub=create_table
              apache-logs not useful
                   "[Thu Mar 19 15:45:21 2009] [notice] child pid 31905 exit signal Bus error (10)"
              but there is a Crashreporter stack trace in Console.app

           Try uncommenting the DEBUG defines in config.php

              @define('DEBUG', true);
              @define('DEBUG_EXTRA', true);


           NB
           
              1)  The stock Sqlite on OSX is not the standard one ... has apple mods 
              2)  testing this on linux non-trivial ... as gave to get PHP
              3)  http://httpd.apache.org/dev/debugging.html

              4)  possibly interference between sqlite versions ... 
                      apache/mod_python/svn/trac are not using the stock sqlite 
                      apache/mod_php/phpbb3  is using sqlite3 command off PATH ?
              5) possibly easier just to go direct to using mysql 


              6) avoid the issues...
                      http://www.apachefriends.org/en/xampp-macosx.html


     phpbb-test
         insert some test php and open it 
         


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

phpbb-base(){    echo $(phpbb-dir)/$(phpbb-pkgd) ; }

phpbb-install(){ phpbb-open install/index.php ; }

phpbb-ln(){
   local iwd=$PWD
   apache-
   cd `apache-htdocs`
   ln -s $iwd/$(phpbb-pkgd) $(phpbb-urlb)
   cd $iwd
}

phpbb-url(){ echo http://localhost/$(phpbb-urlb)/$1 ; }
phpbb-open(){  open $(phpbb-url $*) ; }

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


phpbb-dbpath(){ echo /private/tmp/$FUNCNAME/demo.db ; }

phpbb-db-create(){
  ## this should live in sqlite.bash 
   local dbp=$1
   mkdir -p $(dirname $dbp) 
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
   $SUDO chown -R $(apache-user):$(apache-group) $(dirname $dbp)

}

phpbb-test(){

  local name=t.php
  local tmp=/tmp/$FUNCNAME/$name && mkdir -p $(dirname $tmp)
  phpbb-test- $(phpbb-dbpath) > $tmp
  $SUDO mv $tmp $(phpbb-base)/
  phpbb-open $name
}


phpbb-test-(){

  local dbpath=$1
  cat << EOT
<?php
if (\$db = sqlite_open("$dbpath", 0666, \$sqliteerror)) { 
    sqlite_query(\$db, 'CREATE TABLE foo (bar varchar(10))');
    sqlite_query(\$db, "INSERT INTO foo VALUES ('fnord')");
    \$result = sqlite_query(\$db, 'select bar from foo');
    var_dump(sqlite_fetch_array(\$result)); 
} else {
    die(\$sqliteerror);
}
?>

EOT

}

