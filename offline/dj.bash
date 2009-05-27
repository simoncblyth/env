dj-src(){      echo offline/dj.bash ; }
dj-source(){   echo ${BASH_SOURCE:-$(env-home)/$(dj-src)} ; }
dj-vi(){       vi $(dj-source) ; }
dj-env(){      
   elocal- ; 
   export DJANGO_SETTINGS_MODULE=$(dj-settings-module)
   python- system
}

dj-settings-module(){ echo $(dj-project).settings ; }
dj-urlroot(){         echo /$(dj-project) ; }          

dj-notes(){
  cat << EON

   1) initial investigations on cms01 ... using system python 2.3.2

   2) moved to cms02 when needed to deploy into mod_python
      ... but the apache there is my source apache using my
      source python 2.5. 

      so the prerequsities are less simple 

EON

}


dj-usage(){ 
  cat << EOU
    
     http://www.djangoproject.com
     http://docs.djangoproject.com/en/dev/intro/tutorial01/#intro-tutorial01

     $(env-wikiurl)/MySQL
     $(env-wikiurl)/MySQLPython
     $(env-wikiurl)/OfflineDB


     dj-settings-module : $(dj-settings-module)
         DJANGO_SETTINGS_MODULE : $DJANGO_SETTINGS_MODULE

     dj-env   
         called by the dj- precursor 
         sets up use of system python : required for mysql-python to work on cms01
         due to this it is important to start a new shell before doing "dj-"
         ... supporting env cleanup to avoid this is not worth the effort

     dj-get
     dj-ln
          plant a symbolic link in site-package

     dj-admin
          invoke the django-admin.py


     dj-models-fix
          why is the seqno the primary key needed 
                    ... why was this not introspeced ?


     dj-project       : $(dj-project)
     dj-app           : $(dj-app)

     dj-srcdir        : $(dj-srcdir)
     dj-projdir       : $(dj-projdir)
     dj-appdir        : $(dj-appdir)

     dj-port          : $(dj-port)

     dj-manage <other args>
          invoke the manage.py for the project  $(dj-project) 
          

     dj-run :
     dj-shell :
     dj-syncdb :
            use the django-manage ...    

     dj-cd
          cd to dj-projdir

   TODO :
        fix the names of the model classes ... 
        load the mysqldump with mysqlpython ?

EOU

}

dj-preq(){
    sudo yum install mysql-server

   
    sudo yum install MySQL-python
  
  #   if the system versions dont work ... 
  # pymysql-
  # pymysql-build
  #



  ## this is in dag.repo ... you may need to enable that in /etc/yum.repos.d/dag.repo
    sudo yum install ipython
}


dj-build(){

  local msg="=== $FUNCNAME :"
   dj-get             ## checkout 
   dj-ln              ## plant link in site-packages
   dj-create-db       ## gives error if exists already 

   [ $? -ne 0 ] && echo $msg failed ... probaly you need to : sudo /sbin/service mysqld start && return 1

   ## load from mysqldump 
   offdb-
   offdb-build

   ## introspect the db schema to generate and fix models.py
   dj-models

   dj-ip

}


## src access ##

dj-srcurl(){  echo http://code.djangoproject.com/svn/django/trunk ; }
dj-srcfold(){ echo $(local-base)/env ; }
dj-srcnam(){  echo django ; }
dj-srcdir(){  echo $(dj-srcfold)/$(dj-srcnam) ; }
dj-admin(){   $(dj-srcdir)/django/bin/django-admin.py $* ; }
dj-get(){
  local msg="=== $FUNCNAME :"
  local dir=$(dj-srcfold)
  local nam=$(dj-srcnam)
  mkdir -p $dir && cd $dir 
  [ ! -d "$nam" ] && svn co $(dj-srcurl)  $nam || echo $msg $nam already exists in $dir skipping 
}
dj-ln(){
  local msg="=== $FUNCNAME :"
  python-ln $(dj-srcdir)/django 
  python-ln $(env-home) env
  python-ln $(dj-projdir)
}

dj-find(){
  local q=$1
  local iwd=$PWD
  cd $(dj-srcdir)
  find . -name "*.py" -exec grep -H $1 {} \;
}

## project/app layout ##

dj-project(){ echo ${DJANGO_PROJECT:-dybsite} ; }
dj-app(){     echo ${DJANGO_APP:-offdb} ; }
dj-port(){    echo 8000 ; }
dj-projdir(){ echo $(env-home)/offline/$(dj-project) ; }
dj-appdir(){  echo $(dj-projdir)/$(dj-app) ; }
dj-apppkg(){  echo env.offline.$(dj-project).$(dj-app) ; }
dj-cd(){      cd $(dj-appdir) ; }

## database setup   ##

dj-val(){ echo $(private- ; private-val $1) ;}
dj-create-db(){ echo "create database if not exists $(dj-val DATABASE_NAME) ;"  | dj-mysql- ; }
dj-mysql-(){    mysql --user $(dj-val DATABASE_USER) --password=$(dj-val DATABASE_PASSWORD) $1 ; }
dj-mysql(){     dj-mysql- $(dj-val DATABASE_NAME) ; } 

## models introspection from db ##  

dj-models(){
   local msg="=== $FUNCNAME :"
   echo $msg creating $FUNCNAME-path
   $FUNCNAME-inspectdb
   $FUNCNAME-fix 
}
dj-models-path(){  echo $(dj-appdir)/generated/models.py ; }
dj-models-inspectdb(){
   local path=$(dj-models-path)
   mkdir -p $(dirname $path) 
   touch $(dirname $path)/__init__.py
   dj-manage inspectdb > $path
}
dj-models-fix(){
   ## this may be table specific
   perl -pi -e "s@null=True, db_column='SEQNO', blank=True@primary_key=True, db_column='SEQNO'@ " $(dj-models-path)
}

## interactive access to model objects

dj-ip(){      ipython $(dj-appdir)/imports.py ; }


## management interface  ##

dj-manage(){
   local iwd=$PWD
   cd $(dj-projdir)   
   case $1 in 
       shell) ipython manage.py $* ;;
           *)  python manage.py $* ;;
   esac
   cd $iwd
}
dj-run(){    dj-manage runserver $(dj-port) ; }
dj-shell(){  dj-manage shell  ; }
dj-syncdb(){ dj-manage syncdb ; }


## web interface ##

dj-open(){      open http://localhost:$(dj-port $*) ; }


## deployment  ##

dj-confname(){ echo zdjango.conf ; }
dj-eggcache(){ echo /var/cache/dj ; }
dj-deploy(){
  local msg="=== $FUNCNAME :" 
  local tmp=/tmp/env/dj && mkdir -p $tmp 
  local conf=$tmp/$(dj-confname)
  dj-location- > $conf
  apache-
  cat $conf
  local cmd="sudo cp $conf $(apache-confd)/$(basename $conf)"
  local ans
  read -p "$msg Proceed with : $cmd : enter YES to continue  " ans
  [ "$ans" != "YES" ] && echo $msg skipping && return 0

  eval $cmd

  local cache=$(dj-eggcache)
  echo $msg createing egg cache dir $cache
  sudo mkdir -p $cache
  apache-chown $cache
  sudo chcon -R -t httpd_sys_script_rw_t $cache
  ls -alZ $cache
}

dj-export(){
  python-
  sudo rm $(python-site)/$(dj-project)
  sudo svn export $(dj-projdir) $(python-site)/$(dj-project)
}

dj-location-(){
  cat << EOL
<Location "$(dj-urlroot)/">
    SetHandler python-program
    PythonHandler django.core.handlers.modpython
    SetEnv DJANGO_SETTINGS_MODULE $(dj-settings-module)    
    SetEnv PYTHON_EGG_CACHE $(dj-eggcache)
    PythonPath "['$(dirname $(dj-projdir))', '$(dj-srcdir)'] + sys.path"
    PythonOption django.root $(dj-urlroot)
    PythonDebug On
</Location>
EOL
}

## test ##

dj-audit(){ sudo vi /var/log/audit/audit.log ; }
dj-audit-tail(){ sudo tail -f  /var/log/audit/audit.log ; }
dj-selinux(){
   #sudo vi /var/log/audit/audit.log
   local msg="=== $FUNCNAME :"
   [ "$NODE_TAG" != "N" ] && echo $msg only needed on N && return 1 
   sudo chcon -R -t var_t /data1/env
   sudo chcon -R -t httpd_sys_content_t $(dj-srcdir)
   sudo chcon -R -t httpd_sys_content_t $(dj-projdir) 
}


dj-check-settings(){
   python -c "import dybsite.settings "
}


dj-test(){
    curl http://localhost$(dj-urlroot)/
}



dj-check(){

  python -c "import dybsite.settings as s ; print '\n'.join(['%s : %s ' % ( v, getattr(s, v) ) for v in dir(s) if v.startswith('DATABASE_')]) "

}
