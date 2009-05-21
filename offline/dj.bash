# === func-gen- : offline/dj.bash fgp offline/dj.bash fgn dj
dj-src(){      echo offline/dj.bash ; }
dj-source(){   echo ${BASH_SOURCE:-$(env-home)/$(dj-src)} ; }
dj-vi(){       vi $(dj-source) ; }
dj-env(){      
   elocal- ; 
   export DJANGO_SETTINGS_MODULE=env.offline.$(dj-project).settings
}


dj-usage(){ 
  cat << EOU
    
     http://www.djangoproject.com
     http://docs.djangoproject.com/en/dev/intro/tutorial01/#intro-tutorial01

         DJANGO_SETTINGS_MODULE : $DJANGO_SETTINGS_MODULE

     dj-get
     dj-ln
          plant a symbolic link in site-package

     dj-admin
          invoke the django-admin.py

     dj-startproject name
          create the name project using "django-admin startproject name"

     dj-settings- <name>
          db config in the settings.py $(django-settings-path)
     dj-settings <name>
          edit the settings file for the default project  
     dj-urls <name>
          edit the urls file for the default project  

     dj-port          : $(django-port)
     dj-project       : $(django-project)
     dj-projdir       : $(django-projdir)
     dj-dbpath        : $(django-dbpath)

     dj-manage <other args>
          invoke the manage.py for the project  $(django-project) 
          

     dj-run :
     dj-shell :
     dj-syncdb :
            use the django-manage ...    

     dj-cd
          cd to dj-projdir


EOU

}


## src access ##

dj-srcfold(){ echo $(local-base)/env/dj ; }
dj-srcnam(){  echo django ; }
dj-srcdir(){  echo $(dj-srcfold)/$(dj-srcnam) ; }
dj-admin(){   $(dj-srcdir)/bin/django-admin.py $* ; }
dj-get(){
  local dir=$(dj-srcfold)
  mkdir -p $dir && cd $dir 
  svn co http://code.djangoproject.com/svn/django/trunk $(dj-srcnam)
}
dj-ln(){
  cd
  python-
  local cmd="sudo ln -s $(dj-srcdir) $(python-site)/django"
  echo $cmd
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
dj-cd(){      cd $(dj-appdir) ; }


dj-port(){    
   case ${1:-$(dj-project)} in
     *) echo 8000 ;;
   esac
}

## proj infrastructure creation ##

dj-startproject(){
   local name=$(basename $(dj-projdir))
   cd $(dirname $(dj-projdir))
   [ -d "$name" ] && echo $msg project $name exists already && return 1
   dj-admin startproject  $name
}

## settings ##

dj-dbpath(){  echo $(dj-projdir)/sqlite3.db ; }
dj-settings-(){
   local iwd=$PWD
   local db="sqlite3"
   local dbpath=$(dj-dbpath)
   cd $(dj-projdir)
   perl -pi -e "s,(DATABASE_ENGINE\s*=\s*')(\S*)('.*)$,\$1$db\$3,"   settings.py
   perl -pi -e "s,(DATABASE_NAME\s*=\s*')(\S*)('.*)$,\$1$dbpath\$3," settings.py
   cd $iwd
}
dj-settings(){  vi $(dj-projdir)/settings.py ; }
dj-urls(){      vi $(dj-projdir)/urls.py ; }

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
dj-ip(){      ipython $(dj-projdir)/ienv.py ; }


## web interface ##

dj-open(){      open http://localhost:$(dj-port $*) ; }


## table manipulations ##

dj-table(){ 
   cat << EOT    
mytree_node 
mytree_leaf
EOT
}
dj-lstab-(){ 
   local tab ; for tab in $(dj-table) ; do 
        echo "select * from $tab ; " 
   done 
}
dj-lstab(){ $FUNCNAME- | sqlite3 $(dj-dbpath) ; }

dj-rmtab-(){ 
   local tab ; for tab in $(dj-table) ; do 
      echo "drop table $tab ; " 
   done
}
dj-rmtab(){ $FUNCNAME- | sqlite3 $(dj-dbpath) ; }


dj-retab(){
   dj-rmtab
   dj-manage syncdb
}


