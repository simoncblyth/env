dj-src(){      echo dj/dj.bash ; }
dj-source(){   echo ${BASH_SOURCE:-$(env-home)/$(dj-src)} ; }
dj-dir-(){     echo $(dirname $(dj-source)) ; }
dj-vi(){       vim $(dj-source) ; }
dj-env(){      
   elocal- ; 
   export DJANGO_SETTINGS_MODULE=$(dj-settings-module)
   #export PYTHON_EGG_CACHE=$(dj-eggcache-dir)

   python-
   ## place the desired python to work with in the path
   case $NODE_TAG in
    C) python- source ;;
    *) echo -n ;;
   esac
}

## environment overridible coordinates
## allowing these functions to be used as the basis for whatever django project

dj-dir(){      echo ${DJANGO_DIR:-$(dj-dir-)} ; }
dj-project(){  echo ${DJANGO_PROJECT:-runinfo} ; } 
dj-protocol(){ echo ${DJANGO_PROTOCOL:-scgi} ; } 
dj-app(){      echo ${DJANGO_APP:-run} ; }        
dj-projdir(){  echo ${DJANGO_PROJDIR:-$(dj-dir)/$(dj-project)} ; }

dj-appdir(){  echo $(dj-projdir)/$(dj-app) ; }
dj-cd(){      cd $(dj-projdir) ; }

dj-info(){    env | grep DJANGO_ ;  }

dj-urls(){            vi $(dj-projdir)/urls.py ; }    
dj-settings(){        vi $(dj-projdir)/settings.py ; }
dj-settings-module(){ echo $(dj-project).settings ; }
dj-urlroot(){         echo /$(dj-project) ; }          
dj-test(){            curl http://localhost$(dj-urlroot)/ ; }

dj-settings-check(){ python -c "from django.conf import settings ; print settings, settings.SETTINGS_MODULE " ; }

dj-usage(){ 
  cat << EOU
   
    dj-* functions for django installation, admin and development
    ================================================================

     http://www.djangoproject.com
     http://docs.djangoproject.com/en/dev/intro/tutorial01/#intro-tutorial01

     $(env-wikiurl)/MySQL
     $(env-wikiurl)/MySQLPython

     dj-preq
            install pre-requisites to running django : python + mysql and python bindings 

     dj-preq-yum
     dj-preq-port
     dj-preq-ipkg
            pre-requisite installers for various package managers  
   

     dj-settings-module : $(dj-settings-module)
         DJANGO_SETTINGS_MODULE : $DJANGO_SETTINGS_MODULE

     dj-env   
         called by the dj- precursor 
         sets up use of system python : required for mysql-python to work on cms01
         due to this it is important to start a new shell before doing "dj-"
         ... supporting env cleanup to avoid this is not worth the effort

     dj-get
         get django with subversion trunk checkout  

     dj-srcnam : $(dj-srcnam)
     dj-ln
          plant a symbolic link in site-package
          pointing at the version of django to use

     dj-admin
          invoke the django-admin.py

     dj-startproject projname
          create initial directory for a project, which contains the settings.py that 
          configures the database connection  
          
          projects contain apps which define the models etc..


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

EOU

}


## pre-requisites to running django 

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
    # if the system versions dont work ... (version mismatches etc.. ) see pymysql-
    sudo yum install ipython    ## from dag repo : enable in /etc/yum.repos.d/dag.repo
}

dj-preq-ipkg(){
   sudo ipkg install python25
   # sudo ipkg install py25-django  ... using django trunk is stable enough and more convenient 
}


dj-get-version(){
 python -c "from django import get_version ; print get_version() "
}

dj-versions(){

   dj-get-version

   python-
   python-versions

   mysql-
   mysql-versions

   apache-
   apache-versions

   svn info $(dj-srcdir)
}


dj-build(){
   local msg="=== $FUNCNAME :"

   dj-get             ## checkout 
   dj-ln              ## plant link in site-packages

   djext-             ## utility django ./manage.py extension commands such as : shell_plus
   djext-build

   mysql-
   mysql-create-db    ## mysql must be running
   [ $? -ne 0 ] && echo $msg failed ... probaly you need to : sudo /sbin/service mysqld start && return 1
}


## src access  ... NB the "django" link in python site-packages is the definitive chooser of django version  

dj-srcurl(){  echo http://code.djangoproject.com/svn/django/trunk ; }
dj-srcfold(){ echo $(local-base)/env/dj ; }  ## rename to match this controlling script
dj-srcnam(){  echo django ; }
dj-ls(){      ls -l $(dj-srcfold) ; }
dj-srcdir-(){ echo $(dj-srcfold)/$(dj-srcnam) ; }
dj-srcdir(){  python-rln django ; }                  ## read the link 
dj-scd(){     cd $(dj-srcdir) ; }
dj-mate(){   [ -d "$(dj-srcdir)" ] &&  mate $(dj-srcdir) || echo === $FUNCNAME : error no dj-srcdir:$(dj-srcdir)  ; }
dj-admin(){   $(dj-srcdir)/bin/django-admin.py $* ; }
dj-port(){    echo 8000 ; }

dj-startproject(){    ## command not available if the envvar is defined 
   DJANGO_SETTINGS_MODULE= dj-admin startproject $*
}

## django source ##

dj-get(){
  local msg="=== $FUNCNAME :"
  local dir=$(dj-srcfold) && mkdir -p $dir && cd $dir 
  local nam=$(dj-srcnam)
  [ ! -d "$nam" ] && svn co $(dj-srcurl)  $nam || echo $msg $nam already exists in $dir skipping 
}

dj-update(){ svn up $(dj-srcdir-) ; }

dj-find(){
  local q=$1
  cd $(dj-srcdir)
  find . -name "*.py" -exec grep -H $1 {} \;
}

## hookup with python ##

dj-ln(){
  local msg="=== $FUNCNAME :"
  python-
  python-ln $(dj-srcdir-)/django django   ## set the link
  python-ln $(env-home) env
  python-ln $(dj-projdir)
}

## management interface  ##

dj-manage(){ cd $(dj-projdir) ; python ./manage.py $*  ; }
dj-run(){    dj-manage runserver $(dj-port) ; }
dj-shell(){  dj-manage shell_plus  ; }   ## requires dj-ext 
dj-db(){  dj-manage dbshell  ; } 
dj-syncdb(){ dj-manage syncdb ; }

dj-open(){   open http://localhost:$(dj-port $*) ; }








