# === func-gen- : offline/dj.bash fgp offline/dj.bash fgn dj
dj-src(){      echo offline/dj.bash ; }
dj-source(){   echo ${BASH_SOURCE:-$(env-home)/$(dj-src)} ; }
dj-vi(){       vi $(dj-source) ; }
dj-env(){      
   elocal- ; 
   export DJANGO_SETTINGS_MODULE=env.offline.$(dj-project).settings

   python- system
}


dj-usage(){ 
  cat << EOU
    
     http://www.djangoproject.com
     http://docs.djangoproject.com/en/dev/intro/tutorial01/#intro-tutorial01

     $(env-wikiurl)/MySQL
     $(env-wikiurl)/MySQLPython
     $(env-wikiurl)/OfflineDB


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

     dj-startproject name
          create the name project using "django-admin startproject name"

     dj-setup
          inplace edits entering DATABASE_* coordinates in settings.py 


     dj-settings- <name>
          db config in the settings.py 
     dj-settings-vi 
          edit the settings file for the default project  


     dj-models-fix
          why is the seqno the primary key needed 
                    ... why was this not introspeced ?



     dj-urls <name>
          edit the urls file for the default project  


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


   ISSUES :
      the settings.py contains a mix of 
           * sensitive stuff that should not be kept in a repository 
           * stuff that should be ...

   TODO :
         easy_install + ipython into system python
         automate the __unicode__ generation + avoid Auth* Django* classes     

EOU

}


dj-build(){

   dj-get              ## checkout 
   dj-ln               ## plant link in site-packages

   dj-startproject 

   dj-settings        ## DATABASE_* coordinates in settings.py
   dj-create-db       ## gives error if exists already 

   ## load from mysqldump 
   offdb-
   offdb-load 

   ## introspect the db schema to generate and fix models.py
   dj-models

   ## dump using the django introspected model
   dj-models-dump

   ## dj-manage syncdb ... creates Auth* and Django* tables only needed for web access ? 
}



dj-preq(){
    sudo yum install mysql-server
    sudo yum install MySQL-python
    sudo yum install ipython
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
  python-ln $(env-home)
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
dj-projdir(){ echo $(env-home)/offline/$(dj-project) ; }
dj-appdir(){  echo $(dj-projdir)/$(dj-app) ; }
dj-apppkg(){  echo env.offline.$(dj-project).$(dj-app) ; }
dj-cd(){      cd $(dj-appdir) ; }


dj-port(){    
   case ${1:-$(dj-project)} in
     *) echo 8000 ;;
   esac
}

## proj infrastructure creation ##

dj-startproject(){
   local msg="=== $FUNCNAME :"
   local name=$(basename $(dj-projdir))
   cd $(dirname $(dj-projdir))
   [ -d "$name" ] && echo $msg project $(dj-projdir) exists already && return 0
   echo $msg creating $(dj-projdir)
   dj-admin startproject  $name
}

## settings : NEVER put sensitive things in repository  ##

dj-settings(){
   local msg="=== $FUNCNAME :"
   local path=$($FUNCNAME-path)
   echo $msg editing $path
   dj-settings-apply
   cat $path | grep DATABASE
}

dj-settings-path(){ echo $(dj-projdir)/settings.py ; }
dj-settings-vars(){ echo DATABASE_ENGINE DATABASE_NAME DATABASE_USER DATABASE_PASSWORD DATABASE_HOST ; }
dj-settings-(){
   local var=$1
   local val=$2
   perl -pi -e "s,($var\s*=\s*')(\S*)('.*)$,\$1$val\$3,"   $(dj-settings-path)
}
dj-settings-apply(){
   local msg="=== $FUNCNAME :"
   echo $msg inplace edits of settings.py 
   local var ; for var in $(dj-settings-vars) ; do
      local val=$(dj-settings-val $var)
      printf "%-20s %s \n" $var $val
      dj-settings- $var $val
   done
}

dj-settings-val(){ echo $(private- ; private-val $1) ;}
dj-settings-vi(){  vi $(dj-settings-path) ; }


## urls ##

dj-urls-vi(){      vi $(dj-projdir)/urls.py ; }


## database setup   ##

dj-create-db-(){
cat << EOC
CREATE DATABASE ${1:-dbname} ;
EOC
}
dj-create-db(){ $FUNCNAME- $(dj-settings-val DATABASE_NAME)  | dj-mysql- ; }
dj-mysql-(){  mysql --user $(dj-settings-val DATABASE_USER) --password=$(dj-settings-val DATABASE_PASSWORD) $1 ; }
dj-mysql(){   dj-mysql- $(dj-settings-val DATABASE_NAME) ; } 


## models introspection from db ##  

dj-models(){
   local msg="=== $FUNCNAME :"
   echo $msg creating $FUNCNAME-path
   $FUNCNAME-inspectdb
   $FUNCNAME-fix 
   $FUNCNAME-proxy
}
dj-models-path(){  echo $(dj-appdir)/genmodels.py ; }
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
dj-models-proxy-(){
   cat << EOC
from $(dj-apppkg) import genmodels
from env.offline.dj import ProxyWrap
exec str(ProxyWrap(genmodels))
EOC
}
dj-models-proxy(){
   $FUNCNAME- > $(dj-appdir)/models.py
}

## ... also do imports prep for ipython usage with a spot of code generation 
## ... then minimise what must be updated following a genmodels update 

dj-models-classes(){ cat $(dj-models-path) | perl -n -e 'm,class (\S*)\(models.Model\):, && print "$1\n" ' ; }
dj-models-imports(){ $FUNCNAME- > $(dj-projdir)/$FUNCNAME.py ; }
dj-models-imports-(){
   local cls ; dj-models-classes | while read cls ; do
      echo "from env.offline.$(dj-project).$(dj-app).models import $cls "
   done
}
dj-models-printall-(){
   local cls ; dj-models-classes | while read cls ; do
      echo "for o in $cls.objects.all():print o"
   done
}
dj-models-dump-(){
    dj-models-imports-
    dj-models-printall-
}
dj-models-dump(){ $FUNCNAME- | python ; }





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
dj-ip(){      ipython $(dj-projdir)/dj-models-imports.py ; }


## web interface ##

dj-open(){      open http://localhost:$(dj-port $*) ; }



