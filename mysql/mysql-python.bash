# === func-gen- : mysql/mysql-python fgp mysql/mysql-python.bash fgn mysql-python fgh mysql
mysql-python-src(){      echo mysql/mysql-python.bash ; }
mysql-python-source(){   echo ${BASH_SOURCE:-$(env-home)/$(mysql-python-src)} ; }
mysql-python-vi(){       vi $(mysql-python-source) ; }
mysql-python-env(){      elocal- ; }
mysql-python-usage(){
  cat << EOU
     mysql-python-src : $(mysql-python-src)
     mysql-python-dir : $(mysql-python-dir)




    mysql-python-*
          http://mysql-python.blogspot.com/
               

  == pip install -U mysql-python ==

      installed MySQL-python-1.2.3.tar.gz  after uninstalling 1.2.3c1
      the mysql-config in PATH determined which mysql/python to build against






 == unreleased 1.3.0 incompatible with django ==

    was operating with unreleases 1.3.0 (for cython wrapping ease) 
    but django not compatible with this ...
    attempt to back off to 1.2.3c1 by easy editing 

      File "/data/env/system/python/Python-2.5.1/lib/python2.5/site-packages/django/db/backends/mysql/base.py", line 14, in <module>
    raise ImproperlyConfigured("Error loading MySQLdb module: %s" % e)
    django.core.exceptions.ImproperlyConfigured: Error loading MySQLdb module: libmysqlclient_r.so.15: cannot open shared object file: No such file or directory


     https://mysql-python.svn.sourceforge.net/svnroot/mysql-python/tags/MySQLdb-1.2.3/


== cf svn tagged 1.2.3 with tarball ==

[blyth@cms01 mysql]$ diff -r --brief MySQLdb-1.2.3/MySQLdb/ MySQL-python-1.2.3/ | grep -v .svn
Only in MySQL-python-1.2.3/MySQLdb: release.py
Only in MySQL-python-1.2.3/: MySQL_python.egg-info
Only in MySQL-python-1.2.3/: PKG-INFO
Files MySQLdb-1.2.3/MySQLdb/setup.cfg and MySQL-python-1.2.3/setup.cfg differ 


i


EOU
}





mysql-python-ver(){ echo 1.2.3 ; }
#mysql-python-name(){ echo MySQL-python-$(mysql-python-ver) ; }
#mysql-python-name(){ echo MySQLdb-2.0 ;}
mysql-python-name(){ echo MySQLdb-$(mysql-python-ver) ;}    ## svn checkout of the tag, to facilitate patching 


mysql-python-url(){ 
   local nam=$(mysql-python-name)
   case $nam in  
              MySQLdb-2.0) echo http://mysql-python.hg.sourceforge.net/hgweb/mysql-python/$(mysql-python-name)/   ;;
               MySQLdb-1*) echo https://mysql-python.svn.sourceforge.net/svnroot/mysql-python/tags/MySQLdb-$(mysql-python-ver)/ ;;
            MySQL-python*) echo http://downloads.sourceforge.net/project/mysql-python/mysql-python/$(mysql-python-ver)/$nam.tar.gz ;; 
   esac
}

mysql-python-rdir(){ 
   local nam=$(mysql-python-name)
   case $nam in  
              MySQLdb-1.2.3) echo MySQLdb ;;
                          *) echo -n ;;
   esac
}

mysql-python-dir(){ echo $(local-base)/env/mysql/$(mysql-python-name) ; }
mysql-python-cd(){  cd $(mysql-python-dir)/$(mysql-python-rdir) ; }
mysql-python-mate(){ mate $(mysql-python-dir) ; }
mysql-python-get(){
   local dir=$(dirname $(mysql-python-dir)) &&  mkdir -p $dir && cd $dir
   local nam=$(mysql-python-name)
   case "$nam" in
        MySQLdb-2.0) mysql-python-get-hg  ;;
      MySQLdb-1.2.3) mysql-python-get-svn ;;
                  *) mysql-python-get-tgz ;;
   esac
}

mysql-python-patchpath(){
   echo $(dirname $(mysql-python-source))/$(mysql-python-name)-resultiter.patch 
}

mysql-python-patch(){
   local path=$(mysql-python-patchpath)
   [ ! -f "$path" ] && echo $msg no patch for $(mysql-python-name) && return

   mysql-python-cd
   patch -p0 < $path 
}

mysql-python-makepatch(){
   local msg="=== $FUNCNAME : "
   local path=$(mysql-python-patchpath)
   echo $msg updating $path 
   mysql-python-cd
   svn diff > $path 
}





mysql-python-get-svn(){
  svn co $(mysql-python-url) 
}
mysql-python-get-hg(){
  hg clone $(mysql-python-url) 
}
mysql-python-get-tgz(){
  local tgz=$(mysql-python-name).tar.gz
  [ ! -f "$tgz" ] && curl -L -O $(mysql-python-url)  
  [ ! -d "$nam" ] && tar zxvf $tgz
}

mysql-python-which(){
   which python
   which mysql_config
}

mysql-python-install(){
   mysql-python-cd
   mysql-python-which
   python setup.py install
}

mysql-python-ls(){
  type $FUNCNAME
  ls -1 $(python-site)/MySQL_python*
  grep MySQL $(python-site)/easy-install.pth 
}


mysql-python-wipe(){
  local iwd=$PWD
  python-cd
  rm -rf MySQL_python*
  
  perl  

}




mysql-python-version(){    python -c "import MySQLdb as _ ; print _.__version__ " ; }
mysql-python-installdir(){ python -c "import os,MySQLdb as _ ; print os.path.dirname(_.__file__) " ; }
mysql-python-info(){ cat << EOI
    hostname   : $(hostname)
    version    : $(mysql-python-version)
    installdir : $(mysql-python-installdir)
EOI
}




