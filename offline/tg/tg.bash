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
    
     http://www.djangoproject.com
     http://docs.djangoproject.com/en/dev/intro/tutorial01/#intro-tutorial01

     $(env-wikiurl)/MySQL
     $(env-wikiurl)/MySQLPython
     $(env-wikiurl)/OfflineDB

     tg-env   
         called by the tg- precursor 

     tg-get

     tg-mode   : $(tg-mode)
     tg-srcnam : $(tg-srcnam)
         
     tg-admin

     tg-models-fix
          why is the seqno the primary key needed 
                    ... why was this not introspeced ?


     tg-project       : $(tg-project)
     tg-app           : $(tg-app)

     tg-srcdir        : $(tg-srcdir)
     tg-projdir       : $(tg-projdir)
     tg-appdir        : $(tg-appdir)

     tg-port          : $(tg-port)

     tg-manage <other args>
          invoke the manage.py for the project  $(tg-project) 
          

     tg-run :
     tg-shell :
     tg-syncdb :
     tg-cd
          cd to tg-projdir
EOU

}

tg-preq(){
    local msg="=== $FUNCNAME :"
    [ "$(which port)" != "" ] && $FUNCNAME-port
    [ "$(which yum)"  != "" ] && $FUNCNAME-yum
    echo $msg ERROR ... no distro handler 
}




tg-preq-port(){
    ## port list installed is too slow to use for this
    [ "$(which python)" != "/opt/local/bin/python" ]              && sudo port install python25
    [ "$(which ipython)" != "/opt/local/bin/ipython" ]            && sudo port install py25-ipython -scientific

    [ "$(which mysql5)" != "/opt/local/bin/mysql5" ]              && sudo port install mysql5
    [ ! -f "/opt/local/lib/python2.5/site-packages/_mysql.so" ]   && sudo port install py25-mysql

    [ "$(which apachectl)" != "/opt/local/bin/apachectl" ]        && sudo port install apache2
    [ ! -f "/opt/local/smth" ]                                    && sudo port install mod_python25
}

tg-preq-yum(){

    sudo yum install mysql-server
    sudo yum install MySQL-python
  
  #   if the system versions dont work ... 
  # pymysql-
  # pymysql-build
  #
  ## this is in dag.repo ... you may need to enable that in /etc/yum.repos.d/dag.repo
    sudo yum install ipython
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



