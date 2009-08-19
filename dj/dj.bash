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

dj-projdir(){ echo $(dj-dir)/$(dj-project) ; }
dj-appdir(){  echo $(dj-projdir)/$(dj-app) ; }
dj-cd(){      cd $(dj-projdir) ; }

dj-settings(){      vi $(dj-projdir)/settings.py ; }                                                                                                                                                                                                          
dj-urls(){          vi $(dj-projdir)/urls.py ; }    

dj-settings-module(){ echo $(dj-project).settings ; }
dj-urlroot(){         echo /$(dj-project) ; }          


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



dj-info(){ env | grep DJANGO_ ;  }

dj-versions(){
   python -V
   echo ipython $(ipython -V)
   python -c "import mod_python as _ ; print 'mod_python:%s' % _.version "
   python -c "import MySQLdb as _ ; print 'MySQLdb:%s' % _.__version__ "
   echo "select version() ; " | dj-mysql
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


## cpk fork investigations ##

dj-cpkurl(){  echo git://github.com/dcramer/django-compositepks.git ; }
dj-cpkrev(){  echo 9477 ; }
dj-cpk(){
    local msg="=== $FUNCNAME :"
    local dir=$(dj-srcfold)
    mkdir -p $dir && cd $dir  
   
    local cpk=$(dj-srcnam cpk)
    local pre=$(dj-srcnam pre)
    echo $msg clone/co $cpk and $pre and compare 
    [ ! -d "$cpk" ] && git clone $(dj-cpkurl)
    [ ! -d "$pre" ] && svn co    $(dj-srcurl)@$(dj-cpkrev) $pre

   
}


dj-diff-(){
    local msg="=== $FUNCNAME :"
    local dir=$(dj-srcfold)
    cd $dir
    local aaa=$(dj-srcnam $1)
    local bbb=$(dj-srcnam $2)
    diff -r --brief $aaa $bbb | grep -v .svn | grep -v .pyc | grep Files  
}
dj-diff(){
   $FUNCNAME- $* | while read line ; do
      dj-diff-parse $line $* || return 1
   done
}
dj-diff-parse(){
   [ "$1" != "Files"  ] && return 1
   [ "$3" != "and"    ] && return 2
   [ "$5" != "differ" ] && return 3
   local aaa=$(dj-srcnam $6)/
   local bbb=$(dj-srcnam $7)/
   local a=${2/$aaa/}
   local b=${4/$bbb/}
   [ "$a" != "$b" ]  && return 4 
   echo $a   
}
dj-opendiff(){
   local aaa=$(dj-srcnam $1)/
   local bbb=$(dj-srcnam $2)/
   dj-diff $* | while read line ; do 
      echo opendiff $aaa$line $bbb$line 
   done
}



dj-cpk-mate(){ mate $(dj-srcfold)/$(dj-srcnam ${1:-cpk});  }


dj-git(){

   local dir=$(dj-srcfold)/djgit && mkdir -p $dir
   cd $dir
   [ ! -d .git ]  && git init 

   ## add some aliases to remotes  
   git remote add dcramer  git://github.com/dcramer/django-compositepks.git
   git remote add django   git://github.com/django/django.git

   ## access the urls
   git config --get remote.dcramer.url
   git config --get remote.django.url

   ## using the aliases for the fetch, stores the branches in eg django/master and dcramer/master 
   git fetch django
   git fetch dcramer
   
   ## create local branches to track the remote ones
   git checkout -b django  django/master 
   git checkout -b dcramer dcramer/master 

   ## list all branches , local and remote 
   git branch -a 


}


dj-git-try(){

   local dir=$(dj-srcfold)
   cd $dir 
   [ ! -d "django-compositepks" ] && git clone git://github.com/dcramer/django-compositepks.git
   cd django-compositepks

   git branch aswas    ## for easy switchback

   [ "$(git config --get remote.django.url)" == "" ] && git remote add django git://github.com/django/django.git  

   
   git fetch django 

   echo $msg git merge django/master  ... and fix conflicts 

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
dj-srcdir(){  echo $(dj-srcfold)/$(dj-srcnam) ; }
dj-mate(){    mate $(dj-srcdir) ; }
dj-admin(){   $(dj-srcdir)/django/bin/django-admin.py $* ; }
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



dj-five(){ [ "$(uname)" == "Darwin" ] && echo 5 ; }
dj-val(){ echo $(private- ; private-val $1) ;}
dj-create-db(){ echo "create database if not exists $(dj-val DATABASE_NAME) ;"  | dj-mysql- ; }
dj-mysql-(){    mysql$(dj-five) --user $(dj-val DATABASE_USER) --password=$(dj-val DATABASE_PASSWORD) $1 ; }
dj-mysql(){     dj-mysql- $(dj-val DATABASE_NAME) ; } 

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



dj-models-fix-deprecated(){
   perl -pi -e "s@null=True, db_column='ROW_COUNTER', blank=True@primary_key=True, db_column='ROW_COUNTER'@ " $(dj-models-path)
}


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


## fixtures : allow saving the state of tables ... for reloading on syncdb

dj-initialdata-path(){ echo ${1:-theapp}/fixtures/initial_data.json ;  }
dj-dumpdata-(){
    local app=${1:-dbi}
    local iwd=$PWD
    cd $(dj-projdir)  
    python manage.py dumpdata --format=json --indent 1 $*
    cd $iwd
}
dj-dumpdata(){
    local msg="=== $FUNCNAME :"
    local app=${1:-dbi}
    local path=$(dj-initialdata-path $app)
    local fixd=$(dirname $path)
    [ ! -d "$fixd" ] && echo $msg ABORT no dir $fixd && return 1
    [ -f "$path" ]   && echo $msg ABORT path $path already exists && return 2
    echo $msg app $app writing to $path 
    dj-dumpdata- $app > $path
}
dj-loaddata-(){
    local app=${1:-dbi}
    local iwd=$PWD
    cd $(dj-projdir)  
    python manage.py loaddata $*
    cd $iwd
}
dj-loaddata(){
    local msg="=== $FUNCNAME :"
    local app=${1:-dbi}
    local path=$(dj-initialdata-path $app)
    [ ! -f "$path" ]   && echo $msg ABORT no path $path && return 2
    echo $msg from $path 
    dj-loaddata- $path
}



## web interface ##

dj-open(){      open http://localhost:$(dj-port $*) ; }


## deployment  ##

dj-confname(){ echo 50-django.conf ; }
dj-eggcache-dir(){ 
    case ${USER:-nobody} in 
      nobody|apache|www) echo /var/cache/dj ;; 
                      *) echo $HOME ;;  
    esac
}
dj-deploy(){

   local msg="=== $FUNCNAME :"
   dj-conf
   [ "$?" != "0" ] && echo $msg && return 1

   dj-eggcache
   dj-selinux

   private- 
   private-sync

   dj-docroot-ln

   dj-syncdb
   #dj-test
}

dj-server(){ 
   case ${1:-$NODE_TAG} in
      U|G) echo lighttpd  ;;
        *) echo apache ;;
   esac
}

dj-conf(){
  local msg="=== $FUNCNAME :" 
  local tmp=/tmp/env/dj && mkdir -p $tmp 
  local conf=$tmp/$(dj-confname)
  
  local server=$(dj-server)
  dj-location-$server- > $conf
  $server-
  cat $conf

  local confd=$($server-confd)
  [ ! -d "$confd" ] && echo $msg ABORT there is no confd : $confd  && return 1

  local cmd="sudo cp $conf $confd/$(basename $conf)"
  local ans
  read -p "$msg Proceed with : $cmd : enter YES to continue  " ans
  [ "$ans" != "YES" ] && echo $msg skipping && return 1
  eval $cmd
}

dj-location-apache-(){
  apache-
  private-
  cat << EOL


LoadModule python_module modules/mod_python.so

## each process only servers one request  ... huge performance hit 
## but good for development as means that code changes are immediately reflected 
MaxRequestsPerChild 1

<Location "$(dj-urlroot)/">
    SetHandler python-program
    PythonHandler django.core.handlers.modpython
    SetEnv ENV_PRIVATE_PATH $(USER=$(apache-user) private-path)    
    SetEnv DJANGO_SETTINGS_MODULE $(dj-settings-module)    
    SetEnv PYTHON_EGG_CACHE $(USER=$(apache-user) dj-eggcache-dir)
    PythonOption django.root $(dj-urlroot)
    PythonDebug On
</Location>

<Location "/media">
    SetHandler None
</Location>

<LocationMatch "\.(jpg|gif|png)$">
    SetHandler None
</LocationMatch>



EOL
# PythonPath "['$(dirname $(dj-projdir))', '$(dj-srcdir)'] + sys.path"
}


dj-fcgiroot(){ echo /django.fcgi ; }
dj-location-lighttpd-(){  cat << EOC

fastcgi.server = (
    "$(dj-fcgiroot)" => (
           "main" => (
               "socket" => "$(dj-socket)",
               "check-local" => "disable",
               "allow-x-send-file" => "enable" , 
                      )
                 ),
)

# The alias module is used to specify a special document-root for a given url-subset. 
alias.url += (
           "/media" => "$(python-site)/django/contrib/admin/media",  
)

url.rewrite-once += (
      "^(/media.*)$" => "\$1",
      "^/favicon\.ico$" => "/media/favicon.ico",
      "^/robots\.txt$" => "/robots.txt",
      "^(/.*)$" => "$(dj-fcgiroot)\$1",
)

EOC
}

dj-socket(){  echo /tmp/mysite.sock ; }

dj-runfcgi(){
  local msg="=== $FUNCNAME :"
  cd $(dj-projdir)
  dj-info
  echo $msg $(pwd)
  which python
  python -V


  #local cmd="sudo ENV_PRIVATE_PATH=$HOME/.bash_private python manage.py runfcgi -v 2 debug=true protocol=fcgi socket=$(dj-socket) daemonize=false maxrequests=1 " 
  #local cmd="sudo ENV_PRIVATE_PATH=$HOME/.bash_private python manage.py runfcgi -v 2 debug=true protocol=fcgi socket=$(dj-socket) daemonize=false " 
  #local cmd="sudo python manage.py runfcgi -v 2 debug=true protocol=fcgi socket=$(dj-socket) daemonize=false " 
  local cmd="ENV_PRIVATE_PATH=$HOME/.bash_private python manage.py runfcgi -v 2 debug=true protocol=fcgi socket=$(dj-socket) daemonize=false " 
  echo $cmd 
  eval $cmd
}

dj-runserver(){
  cd $(dj-projdir)
  ENV_PRIVATE_PATH=$HOME/.bash_private python manage.py runserver 
}







dj-eggcache(){
   local cache=$(dj-eggcache-dir)
   [ "$cache" == "$HOME" ] && echo $msg cache is HOME:$HOME skipping && return 0

   echo $msg createing egg cache dir $cache
   sudo mkdir -p $cache
   apache- 
   apache-chown $cache
   sudo chcon -R -t httpd_sys_script_rw_t $cache
   ls -alZ $cache
}

dj-selinux(){
local msg="=== $FUNCNAME :"

sudo chcon -R -t httpd_sys_content_t $(dj-srcdir)
sudo chcon -R -t httpd_sys_content_t $(dj-projdir) 
sudo chcon -R -t httpd_sys_content_t $(env-home)
}


dj-check-settings(){
type $FUNCNAME
apache-
## python -c "import dybsite.settings " should fail with permission denied 

sudo -u $(apache-user) python -c "import dybsite.settings "
sudo -u $(apache-user)  python -c "import dybsite.settings as s ; print '\n'.join(['%s : %s ' % ( v, getattr(s, v) ) for v in dir(s) if v.startswith('DATABASE_')]) "
}


dj-docroot-ln(){
   local msg="=== $FUNCNAME :"
   apache-
   local docroot=$(apache-docroot)
   local cmd="sudo ln -sf  $(dj-srcdir)/django/contrib/admin/media $docroot/media"
   echo $msg $cmd
   eval $cmd
}


## admin site grabs 

dj-admin-cp(){
  local msg="=== $FUNCNAME :"
  local rel=${1:-templates/admin/base_site.html}
  local srcd=$(dj-srcdir)/django/contrib/admin
  local dstd=$(dj-projdir)  

  echo $msg rel $rel srcd $srcd dstd $dstd 
  local path=$srcd/$rel
  local targ=$dstd/$rel
  [ ! -f "$path" ] && echo $msg ABORT no path $path && return 1
  local cmd="mkdir -p $(dirname $targ) &&  cp $path $targ "
    
  echo $msg $(dirname $path)
  ls -l $(dirname $path)
  echo $msg $(dirname $targ)
  ls -l $(dirname $targ)

  local ans
  read -p "$msg $cmd ... enter YES to proceed " ans
  [ "$ans" != "YES" ] && echo $msg skipping && return 0
  eval $cmd 
}

dj-startproject(){
   ## curiously command not available if the envvar is defined 
   DJANGO_SETTINGS_MODULE= dj-admin startproject $*
}

## test ##


dj-test(){
    curl http://localhost$(dj-urlroot)/
}



