dj-src(){      echo dj/dj.bash ; }
dj-source(){   echo ${BASH_SOURCE:-$(env-home)/$(dj-src)} ; }
dj-dir-(){     echo $(dirname $(dj-source)) ; }
dj-vi(){       vim $(dj-source) ; }
dj-env(){      
   elocal- ; 
   export DJANGO_SETTINGS_MODULE=$(dj-settings-module)
   export PYTHON_EGG_CACHE=$(dj-eggcache-dir)
  
   private- 
   apache- system
   python- system
}

## environment overridible coordinates
dj-dir(){     echo ${DJANGO_DIR:-$(dj-dir-)} ; }
dj-project(){ echo ${DJANGO_PROJECT:-dybsite} ; } ## djsa
dj-app(){     echo ${DJANGO_APP:-offdb} ; }       ## blog 
dj-info(){    env | grep DJANGO_ ;  }


dj-projdir(){ echo $(dj-dir)/$(dj-project) ; }
dj-appdir(){  echo $(dj-projdir)/$(dj-app) ; }
dj-cd(){      cd $(dj-projdir) ; }

dj-urls(){          vi $(dj-projdir)/urls.py ; }    
dj-settings(){      vi $(dj-projdir)/settings.py ; }
dj-settings-module(){ echo $(dj-project).settings ; }
dj-urlroot(){         echo /$(dj-project) ; }          
dj-test(){ curl http://localhost$(dj-urlroot)/ ; }

dj-settings-check(){ python -c "from django.conf import settings ; print settings, settings.SETTINGS_MODULE " ; }

dj-notes(){
  cat << EON

   Proxying was used in order to simply keep generated 
   model files separate from the tweaked other files 

   NB 
     Override the locations with envvars 
          DJANGO_DIR     : $DJANGO_DIR
          DJANGO_APP     : $DJANGO_APP
          DJANGO_PROJECT : $DJANGO_PROJECT

   Deployments 

           http://belle7.nuu.edu.tw/dybsite/admin/
        N   : system python 2.4, mysql 5.0.24, MySQL_python-1.2.2, 
              system Mod Python , apache


           http://cms01.phys.ntu.edu.tw/dybsite/admin/
        C   : system python 2.3, mysql 4.1.22, MySQL_python  
           
               ===> admin pw needs resetting ...


        C2  :
           EXCLUDE FOR NOW AS PRIME REPO SERVER
                 which still uses source python 2.5
                 and source apache 2.0.63


        H :
            ancient machine ... not worth bothering with 


        G   :  
           
             port installed mysql 5.0.67

             Darwin difficulties ... need to be careful with system python  
             do i want to port install python ?

EON
}


dj-versions(){
   python -V
   echo ipython $(ipython -V)
   python -c "import mod_python as _ ; print 'mod_python:%s' % _.version "
   python -c "import MySQLdb as _ ; print 'MySQLdb:%s' % _.__version__ "
   echo "select version() ; " | mysql-sh
   mysql_config --version 
   apachectl -v
   svn info $(dj-srcdir)
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

     dj-mode   : $(dj-mode)
     dj-srcnam : $(dj-srcnam)
     dj-ln
          plant a symbolic link in site-package
          pointing at the version of django + dybsite + .. to use
          
          provides easy way to try out different versions ... simply change
          the dj-mode and rerun dj-ln to switch : no need to stop/start
          apache (when using MaxRequestsPerChild 1)
          

     dj-admin
          invoke the django-admin.py

     dj-startproject projname
          create initial directory for a project, which contains the settings.py that 
          configures the database connection  
          
          projects contain apps which define the models etc..

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

            Create db tables corresponding to models such as 
            the django_* and auth_* tables used for the admin 

            On first run (when no users defined) will be asked :
               You just installed Django's auth system, which means you don't have any superusers defined.
               Would you like to create one now? (yes/no): yes
               Username (Leave blank to use 'blyth'): 
               E-mail address: blyth@hep1.phys.ntu.edu.tw
               Password: 
               Password (again): 
               Superuser created successfully.
               Installing index for admin.LogEntry model
               Installing index for auth.Permission model
               Installing index for auth.Message model


     dj-cd
          cd to dj-projdir




     dj-initialdata-path app   : $(dj-initialdata-path app)  
               
     dj-dumpdata- app
             dump table serialization to stdout 
     
     dj-dumpdata  app
             write table serialization to standard initialdata path

     dj-loaddata  appname
             load table serialization into db
             CAUTION : this is done automatically on doing syncdb

EOU

}

dj-preq(){
    local msg="=== $FUNCNAME :"
    [ "$(which port)" != "" ] && $FUNCNAME-port
    [ "$(which yum)"  != "" ] && $FUNCNAME-yum
    [ "$(which ipkg)"  != "" ] && $FUNCNAME-yum
    echo $msg ERROR ... no distro handler 
}


dj-preq-port(){
    ## port list installed is too slow to use for this
    [ "$(which python)" != "/opt/local/bin/python" ]              && sudo port install python25
    [ "$(which ipython)" != "/opt/local/bin/ipython" ]            && sudo port install py25-ipython -scientific

    [ "$(which mysql5)" != "/opt/local/bin/mysql5" ]              && sudo port install mysql5
    [ ! -f "/opt/local/lib/python2.5/site-packages/_mysql.so" ]   && sudo port install py25-mysql

    [ "$(which apachectl)" != "/opt/local/bin/apachectl" ]        && sudo port install apache2
    [ ! -f "/opt/local/smth" ]                                    && sudo port install mod_python25

    sudo port install py25-docutils   ## for /admin/doc/
}

dj-preq-yum(){

    sudo yum install mysql-server
    sudo yum install MySQL-python
  
  #   if the system versions dont work ... 
  # pymysql-
  # pymysql-build
  #
  ## this is in dag.repo ... you may need to enable that in /etc/yum.repos.d/dag.repo
    sudo yum install ipython
}

dj-preq-ipkg(){
   sudo ipkg install python25
  # sudo ipkg install py25-django
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

   dj-ip-

}


## src access ##


dj-srcurl(){  echo http://code.djangoproject.com/svn/django/trunk ; }
dj-srcfold(){ 
   case $(dj-mode $*) in 
      system) echo $(python-site) ;; 
           *) echo $(local-base)/env/django ;; 
   esac
}
dj-mode(){ 
   case $NODE_TAG in 
     Z) echo system ;;
     G) echo dev ;;
     *) echo def ;;
   esac
}
dj-srcnam(){  
   case ${1:-$(dj-mode)} in
    cpk) echo django-compositepks ;;
    pre) echo django$(dj-cpkrev)   ;;
def|dev) echo django ;;
    svn) echo django ;;
    git) echo djgit  ;;
 system) echo django ;;
      *) echo django ;;
   esac 
}


dj-ls(){      ls -l $(dj-srcfold) ; }
dj-srcdir-(){  echo $(dj-srcfold)/$(dj-srcnam) ; }
dj-srcdir(){  python-rln django ; }                  ## read the link 
dj-scd(){     cd $(dj-srcdir) ; }
dj-mate(){    mate $(dj-srcdir) ; }
dj-admin(){   $(dj-srcdir)/bin/django-admin.py $* ; }
dj-get(){
  local msg="=== $FUNCNAME :"
  [ "$(dj-mode)" == "system" ] && echo $msg system django && return 1
  local dir=$(dj-srcfold)
  local nam=$(dj-srcnam default)
  mkdir -p $dir && cd $dir 
  [ ! -d "$nam" ] && svn co $(dj-srcurl)  $nam || echo $msg $nam already exists in $dir skipping 
}


dj-ln(){
  local msg="=== $FUNCNAME :"

  [ "$(dj-mode)" != "system" ] && python-ln $(dj-srcfold)/djgit/django django 
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

dj-port(){    echo 8000 ; }
## database setup   ##

dj-create-db(){ echo "create database if not exists $(private-;private-val DATABASE_NAME) ;"  | mysql-sh- ; }

## models introspection from db ##  

dj-models(){
   local msg="=== $FUNCNAME :"
   echo $msg creating $($FUNCNAME-path)
   local path=$(dj-models-path)
   mkdir -p $(dirname $path) 
   touch $(dirname $path)/__init__.py
   $FUNCNAME-inspectdb > $path
}
dj-models-path(){  echo $(dj-appdir)/generated/models.py ; }
dj-models-inspectdb(){ dj-manage- inspectdb | python $(dj-dir)/fix.py ; }

## interactive access to model objects

dj-ip-(){
  ipython $(dj-appdir)/imports.py 
}

dj-ip(){     
  local msg="=== $FUNCNAME :"
  apache-
  local user=$(apache-user)
  local home=/tmp/env/$FUNCNAME/$user
  echo $msg $user $home
  mkdir -p $home
  apache-chown $home
  sudo -u $user HOME=$home $(dj-env-inline) ipython $(dj-appdir)/imports.py 
}

## management interface  ##

dj-env-inline(){  echo DJANGO_SETTINGS_MODULE=$(dj-settings-module) PYTHON_EGG_CACHE=$(dj-eggcache-dir) ; }

dj-manage-(){ python $(dj-projdir)/manage.py $*  ; }
dj-manage(){
   local iwd=$PWD
   cd $(dj-projdir)   
   case $1 in 
       shell)  sudo -u $(apache-user) $(dj-env-inline) ipython manage.py $* ;;
           *)  sudo -u $(apache-user) $(dj-env-inline)  python manage.py $* ;;
   esac
   cd $iwd
}
dj-run(){    dj-manage runserver $(dj-port) ; }
dj-shell(){  dj-manage shell  ; }
dj-syncdb(){ 
   local msg="=== $FUNCNAME :"
   echo $msg 
   dj-manage- syncdb 
}

dj-open(){      open http://localhost:$(dj-port $*) ; }

dj-runserver(){
  cd $(dj-projdir)
  ENV_PRIVATE_PATH=$HOME/.bash_private python manage.py runserver 
}

dj-startproject(){    ## curiously command not available if the envvar is defined 
   DJANGO_SETTINGS_MODULE= dj-admin startproject $*
}






