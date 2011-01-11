# === func-gen- : mysql/mysql-python fgp mysql/mysql-python.bash fgn mysql-python fgh mysql
mysql-python-src(){      echo mysql/mysql-python.bash ; }
mysql-python-source(){   echo ${BASH_SOURCE:-$(env-home)/$(mysql-python-src)} ; }
mysql-python-vi(){       vi $(mysql-python-source) ; }
mysql-python-env(){      elocal- ; }
mysql-python-usage(){
  cat << EOU
     mysql-python-src : $(mysql-python-src)
     mysql-python-dir : $(mysql-python-dir)

  == REFERENCES ==

    mysql-python-*
          http://mysql-python.blogspot.com/
 
  == PREREQUISITES ==

      python
      setuptools
      mysql


 == DBG BUILD ==

     python setup.py build_ext --debug --optimize 0
     python setup.py install


     OSX error ... failed to convert type 246  ...  null dtype descr 


 ==  WARNING : KEEPING mysql_numpy DEVELOPMENTS IN SVN DERIVED PATCH  ==
        
        * MUST "svn add ... "  ALL ADDITIONAL FILES TO MYSQL-PYTHON SVN
         (DESPITE CANNOT COMMIT) ON EACH DEV NODE ... 
          OTHERWISE WILL LOOSE MOST OF PATCH 

        * this keeps svn diff and patches in sync between nodes.


             g4pb:MySQLdb blyth$ svn add test.py mysql_numpy.h Makefile 
                        mysql-python-makepatch
              e > svn diff .... ensure the mysql-python patch doesnt loose mods 

             
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

== on OSX(macports) add symbolic link from mysql_config5 to mysql_config ==


== MOVE MANAGEMENT OF PATCH INTO GITHUB ==

    svn export https://mysql-python.svn.sourceforge.net/svnroot/mysql-python/tags/MySQLdb-1.2.3/ mysql_numpy
    cd mysql_numpy
 
    git init
    git add MySQLdb
    git commit -m "original mysql-python 1.2.3 obtained from https://mysql-python.svn.sourceforge.net/svnroot/mysql-python/tags/MySQLdb-1.2.3/  at revision 650 "
 
    git remote add origin git@github.com:scb-/mysql_numpy.git
    git push origin master
           ## need to enter the C id_rsa passphrase here 

    cp ../README .
    git add README
 
    cd MySQLdb
    patch -p0 < /data/env/local/env/home/mysql/MySQLdb-1.2.3-resultiter.patch
 
    git diff
    git status

    git add Makefile
    git add mysql_numpy.h 
    git add test.py
    git add _mysql.c
    git add setup_posix.py
 
    git status
    git push origin master
           ## need to enter the C id_rsa passphrase here 

   
== GETTING FROM GIT ==

   git clone git://github.com/scb-/mysql_numpy.git


== Add public keys to git ==

   * https://github.com/account


== PUSHING FROM ANOTHER NODE ==

  If you get ...

  simon:mysql_numpy blyth$ git push origin master
fatal: remote error: 
  You can't push to git://github.com/scb-/mysql_numpy.git
  Use git@github.com:scb-/mysql_numpy.git


  Then you cloned from a readonly URL ... OR 
  you need to add the relevant public keys to account thru github web UI


== mysql_numpy ==

   google:"mysql python numpy" ...

   adustman on numpy and mysql-python from 2000
       http://mail.scipy.org/pipermail/numpy-discussion/2000-April/000129.html

== build issues on N ==

=== The version info needs to be quoted ===

   Reproducible with 
        make 
{{{
gcc: _mysql.c
sh: -c: line 0: syntax error near unexpected token `('
sh: -c: line 0: `( gcc -pthread -fno-strict-aliasing -DNDEBUG -O2 -g -pipe -Wall -Wp,-D_FORTIFY_SOURCE=2 -fexceptions -fstack-protector --param=ssp-buffer-size=4 -m32 -march=i386 -mtune=generic -fasynchronous-unwind-tables -D_GNU_SOURCE -fPIC -fPIC -Dversion_info=(1,2,3,'final',0) -D__version__=1.2.3 -I/usr/include/mysql -I/usr/lib/python2.4/site-packages/numpy/core/include -I/usr/include/python2.4 -c _mysql.c -o build/temp.linux-i686-2.4/_mysql.o -g -pipe -Wp,-D_FORTIFY_SOURCE=2 -fexceptions -fstack-protector --param=ssp-buffer-size=4 -m32 -fasynchronous-unwind-tables -D_GNU_SOURCE -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -fno-strict-aliasing -fwrapv ; echo $? > /tmp/tmpZj36Dj/4VfOrA ) 2>&1 | tee /tmp/tmpZj36Dj/hRNszn '
_exec_command_posix failed (status=512)
}}}

     The version info needs to be quoted to get past compiler ...
           -Dversion_info=(1,2,3,'final',0) 
           -Dversion_info="(1,2,3,'final',0)" 


=== preinstalled old numpy ... need to upgrade or avoid conflict ===






EOU
}

mysql-python-ver(){ echo 1.2.3 ; }
#mysql-python-name(){ echo MySQL-python-$(mysql-python-ver) ; }
#mysql-python-name(){ echo MySQLdb-2.0 ;}
#mysql-python-name(){ echo MySQLdb-$(mysql-python-ver) ;}    ## svn checkout of the tag, to facilitate patching 
mysql-python-name(){  
   case $USER in 
      #thho) echo MySQLdb-1.2.3 ;; 
      thho) echo mysql_numpy_ro  ;; 
     blyth) echo mysql_numpy ;;            ##  github managed version of patched mysql-python 1.2.3 
  esac
 }       

mysql-python-url(){ 
   local nam=$(mysql-python-name)
   case $nam in  
              mysql_numpy_ro) echo git://github.com/scb-/mysql_numpy.git ;;
              mysql_numpy) echo git@github.com:scb-/mysql_numpy.git ;;
              MySQLdb-2.0) echo http://mysql-python.hg.sourceforge.net/hgweb/mysql-python/$(mysql-python-name)/   ;;
               MySQLdb-1*) echo https://mysql-python.svn.sourceforge.net/svnroot/mysql-python/tags/MySQLdb-$(mysql-python-ver)/ ;;
            MySQL-python*) echo http://downloads.sourceforge.net/project/mysql-python/mysql-python/$(mysql-python-ver)/$nam.tar.gz ;; 
   esac
}
mysql-python-rdir(){ 
   local nam=$(mysql-python-name)
   case $nam in  
     MySQLdb-1.2.3|mysql_numpy*) echo MySQLdb ;;
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
       mysql_numpy*) mysql-python-get-git ;;
                  *) mysql-python-get-tgz ;;
   esac
}

mysql-python-get-svn(){ svn co $(mysql-python-url) ; }
mysql-python-get-hg(){   hg clone $(mysql-python-url) ; }
mysql-python-get-git(){ git clone $(mysql-python-url) $(mysql-python-name) ; }
mysql-python-get-tgz(){
  local tgz=$(mysql-python-name).tar.gz
  [ ! -f "$tgz" ] && curl -L -O $(mysql-python-url)  
  [ ! -d "$nam" ] && tar zxvf $tgz
}

mysql-python-which(){
   which python
   python -V
   which mysql_config
}

mysql-python-fetch(){
   mysql-python-cd
   git fetch     ## pulling is not a good idea, as does merge also ... better to do that manually
}
mysql-python-push(){
   mysql-python-cd
   git push origin master
}
mysql-python-install(){
   mysql-python-cd
   mysql-python-which
   python setup.py install 
}
mysql-python-sinstall(){
   mysql-python-cd
   mysql-python-which
   sudo python setup.py install 
}
mysql-python-build(){
   mysql-python-cd
   mysql-python-which
   python setup.py build 
}

mysql-python-ls(){
  type $FUNCNAME
  ls -1 $(python-site)/MySQL_python*
  grep MySQL $(python-site)/easy-install.pth 
}

mysql-python-wipe(){
  local msg="=== $FUNCNAME :"
  local dir=$(dirname $(mysql-python-dir)) &&  mkdir -p $dir && cd $dir
  local name=$(mysql-python-name)
  [ ! -d "$name" ]  && echo $msg no dir $name  && return 0 
  [ ${#name} -lt 3 ] && echo $msg sanity check failed && return 1
  
  local cmd="rm -rf $name "
  local ans
  read -p "$msg enter YES to proceed with : $cmd " ans 
  [ "$ans" != "YES" ] && echo $msg skipping && return 0
  eval $cmd

}

mysql-python-uninstall(){
  local iwd=$PWD
  python-cd
  rm -rf MySQL_python*
  echo need to uneasy too 
}

mysql-python-version(){    python -c "import MySQLdb as _ ; print _.__version__ " ; }
mysql-python-installdir(){ python -c "import os,MySQLdb as _ ; print os.path.dirname(_.__file__) " ; }
mysql-python-info(){ cat << EOI
    hostname   : $(hostname)
    version    : $(mysql-python-version)
    installdir : $(mysql-python-installdir)
EOI
}







### MOVED AWAY FROM SVN-PATCH MANAGEMENT TO USING GITHUB

mysql-python-patchpath(){ echo $(dirname $(mysql-python-source))/$(mysql-python-name)-resultiter.patch ; }
mysql-python-patch(){
   local path=$(mysql-python-patchpath)
   [ ! -f "$path" ] && echo $msg no patch for $(mysql-python-name) && return
   mysql-python-cd
   patch -p0 < $path 
   svn add  mysql_numpy.h Makefile test.py   
   echo $msg checking consistency between the patched and svn-added WC from mysql-pythin  and the patch from env
   mysql-python-makepatch
   svn diff $(mysql-python-patchpath)
}

mysql-python-makepatch(){
   local msg="=== $FUNCNAME : "
   local path=$(mysql-python-patchpath)
   echo $msg updating $path 
   mysql-python-cd
   svn diff > $path 
}

mysql-python-revert(){
    local msg="=== $FUNCNAME :"
    echo $msg WARNING this reverts current changes and removes added files  ... LOCAL CHANGES NOT IN THE PATCH WILL BE LOST
    local ans
    read -p "$msg enter YES to proceed " ans
    [ "$ans" != "YES" ] && echo $msg OK skipping && return 0
    mysql-python-cd
    svn rm --force mysql_numpy.h test.py Makefile
    svn revert _mysql.c setup_posix.py
    svn st  
}

