tg-src(){      echo offline/tg/tg.bash ; }
tg-source(){   echo ${BASH_SOURCE:-$(env-home)/$(tg-src)} ; }
tg-dir(){      echo $(dirname $(tg-source)) ; }
tg-vi(){       vi $(tg-source) ; }
tg-env(){      
   elocal- ; 
   private- 
   apache- system
   python- system
}

tg-urlroot(){         echo /$(tg-project) ; }          

tg-notes(){
  cat << EON

   Needs python 2.4:2.6 so for sys python are restricted to N  

           http://belle7.nuu.edu.tw/dybsite/admin/
        N   : system python 2.4, mysql 5.0.24, MySQL_python-1.2.2, 
              system Mod Python , apache

EON

}


tg-versions(){
   python -V
   echo ipython $(ipython -V)
   python -c "import mod_python as _ ; print 'mod_python:%s' % _.version "
   python -c "import MySQLdb as _ ; print 'MySQLdb:%s' % _.__version__ "
   echo "select version() ; " | tg-mysql
   mysql_config --version 
   apachectl -v
   svn info $(tg-srcdir)
}

tg-usage(){ 
  cat << EOU
   
     http://www.turbogears.org/2.0/docs/main/DownloadInstall.html

EOU

}

tg-preq(){
    
    python-
    [ "$(python-version)"     != "2.4.3" ]  && echo $msg untested python version && return 1

    setuptools-
    [ "$(setuptools-version)" != "0.6c9" ]  && setuptools-get




}


tg-build(){

  local msg="=== $FUNCNAME :"
   tg-get             ## checkout 
   tg-ln              ## plant link in site-packages
   tg-create-db       ## gives error if exists already 

   [ $? -ne 0 ] && echo $msg failed ... probaly you need to : sudo /sbin/service mysqld start && return 1

   ## load from mysqldump 
   offdb-
   offdb-build

   ## introspect the db schema to generate and fix models.py
   tg-models

   tg-ip-

}



## src access ##

tg-srcurl(){  echo http://code.tgangoproject.com/svn/tgango/trunk ; }
tg-srcfold(){ echo $(local-base)/env ; }
tg-mode(){ echo def ; }
tg-srcdir(){  echo $(tg-srcfold)/$(tg-srcnam) ; }
tg-admin(){   $(tg-srcdir)/tgango/bin/tgango-admin.py $* ; }
tg-get(){
  local msg="=== $FUNCNAME :"
  local dir=$(tg-srcfold)
  local nam=$(tg-srcnam default)
  mkdir -p $dir && cd $dir 
  [ ! -d "$nam" ] && svn co $(tg-srcurl)  $nam || echo $msg $nam already exists in $dir skipping 
}
tg-ln(){
  local msg="=== $FUNCNAME :"
  python-ln $(tg-srcdir)/tgango tgango 
  python-ln $(env-home) env
  python-ln $(tg-projdir)
}

tg-find(){
  local q=$1
  local iwd=$PWD
  cd $(tg-srcdir)
  find . -name "*.py" -exec grep -H $1 {} \;
}



