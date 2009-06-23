tg-src(){      echo offline/tg/tg.bash ; }
tg-source(){   echo ${BASH_SOURCE:-$(env-home)/$(tg-src)} ; }
tg-dir(){      echo $(env-home $*)/$(dirname $(tg-src)) ; }
tg-vi(){       vi $(tg-source) ; }
tg-env(){      
   elocal- ; 
   private- 
   apache- system
   python- system
}

tg-notes(){
  cat << EON
   Needs python 2.4:2.6 so for sys python are restricted to N  

           http://belle7.nuu.edu.tw/dybsite/admin/
        N   : system python 2.4, mysql 5.0.24, MySQL_python-1.2.2, 
              system Mod Python , apache
EON
}

tg-usage(){ 
  cat << EOU
     http://www.turbogears.org/2.0/docs/main/DownloadInstall.html
     http://www.turbogears.org/2.0/docs/main/QuickStart.html
     http://www.voidspace.org.uk/python/configobj.html

     NB have to avoid commiting development.ini 

     tg-install
            create a virtualenv and install tg2 into it (a boatload of circa 20 dependencies)
            ... also modwsgideploy  


     tg-quickstart
             create a project and add to svn with appropriate ignores 
             and skipping of things with sensitive items
              
     tg-wipe  
              delete a quickstarted project 


EOU
}


tg-preq-install(){
   easy_install MySQL-python
}

tg-preq(){
    local msg="=== $FUNCNAME :"
    python-
    [ "$(python-version)"     != "2.4.3" ]  && echo $msg untested python version && return 1
    setuptools-
    [ "$(setuptools-version)" != "0.6c9" ]  && echo $msg no setuptools && return 1
    virtualenv-
    [ "$(virtualenv-version)" != "1.3.3" ] && echo $msg untested virtualenv  && return 1


    ## my additions ... 
    configobj-
    [ "$(configobj-version)" != "4.5.3" ] && echo $msg untested configobj && return 1
    ipython-
    [ "$(ipython-version)" != "0.9.1" ] && echo $msg untested ipython && return 1

    modwsgi-
    [ ! -f "$(modwsgi-so)" ] && echo $msg modwsgi is not present && return 1 
}

tg-activate(){
   local act=$(tg-srcdir)/bin/activate 
   [ -f "$act" ] && . $act 
 }

tg-install(){
   local dir=$(tg-srcdir)
   local fld=$(dirname $dir)
   local nam=$(basename $dir)
   mkdir -p $fld && cd $fld
   virtualenv --no-site-packages $nam
   tg-activate
   cd $(tg-srcdir)
   easy_install -i http://www.turbogears.org/2.0/downloads/current/index tg.devtools
   easy_install modwsgideploy

   python-ln $(env-home) env   ## for env.base.private.Private access

}

tg-deploy-conf-(){
   ## needed for virtual env usage 
   echo "WSGIPythonHome $(tg-srcdir)"
}

tg-deploy(){
   tg-cd

   ## creates the apache folder in the app folder .. used as examples only 
   paster modwsgi_deploy 
   
   apache-
   apache-eggcache /var/cache/tg

   tg-wsgi-deploy
}


tg-eggcache-dir(){
   case ${USER:-nobody} in 
      nobody|apache|www) echo /var/cache/tg ;;
                      *) echo $HOME ;;
    esac
}


tg-selinux(){
   local msg="=== $FUNCNAME :"
   local cmd="sudo chcon -h -R -t httpd_sys_content_t $(tg-srcdir) "
   echo $cmd
   eval $cmd
}

tg-datadir(){
   local dir=$(tg-projdir)/data
   mkdir -p $dir
   apache-
   apache-chown $dir -R
}


tg-ip(){
  local iwd=$PWD
  tg-activate
  tg-cd
  paster shell $(tg-ini)
  cd $iwd
}




tg-wsgi-deploy-(){  cat << EOS

#  $FUNCNAME 
#     http://code.google.com/p/modwsgi/wiki/VirtualEnvironments

ALLDIRS = ["$(tg-srcdir)/lib/python$(python-major)/site-packages"]

import sys
import site

prev_sys_path = list(sys.path)

for dir in ALLDIRS:
    site.addsitedir(dir)

new_sys_path = []
for item in list(sys.path):
    if item not in prev_sys_path:
        new_sys_path.append(item)
        sys.path.remove(item)

sys.path[:0] = new_sys_path 

import os
os.environ['PYTHON_EGG_CACHE'] = '/var/cache/tg'

from paste.deploy import loadapp
application = loadapp('config:$(tg-projdir)/development.ini')

EOS
}

tg-wsgi-deploy(){
  local msg="=== $FUNCNAME :"
  local tmpd=/tmp/env/$FUNCNAME && mkdir -p $tmpd
  local tmp=$tmpd/$(tg-projname).wsgi
  echo $msg writing $tmp
  $FUNCNAME- > $tmp
  modwsgi-
  modwsgi-deploy $tmp 
}


tg-find(){
  find $(tg-srcdir) -name '*.py' -exec grep -H $1 {} \;
}


tg-srcfold(){  echo $(local-base $*)/env ; }
tg-srcnam(){   echo tg2env ; }
tg-srcdir(){   echo $(tg-srcfold $*)/$(tg-srcnam) ; }
tg-projname(){ echo OfflineDB ; }
tg-projdir(){  echo $(tg-dir)/$(tg-projname) ; }


tg-wipe(){
   local msg="=== $FUNCNAME :"
   cd $(tg-dir)
   local proj=$(tg-projname)
   [ ! -d "$proj" ] && echo $msg dir $proj does not exist && return 1 
    
   local cmd="sudo rm -rf $proj"
   local ans 
   read -p "$msg enter YES to proceed with $cmd " ans
   [ "$ans" != "YES" ] && echo $msg skipping && return 0
   eval $cmd

   ## tis important to remove remnants from the parent dir to allow subseqent quickstarting 
   svn revert $proj
}

tg-quickstart(){

   local msg="=== $FUNCNAME :"
   tg-activate
   cd $(tg-dir)

   local proj=$(tg-projname)
   [ -d "$proj" ] && echo $msg dir $proj exists already && return 1 
   paster quickstart --auth --noinput $(tg-projname)

   cd $(tg-projdir)
   python setup.py develop --uninstall
   python setup.py develop

   ## customize the ini with DATABASE_URL + ...
   tg-conf   

   ## labelling and ownership
   tg-selinux
   tg-datadir

   ## add to svn with appropriate ignores and skips
   tg-svn-add
}

tg-svn-ignore-(){ cat << EOI
data
*.ini
EOI
}

tg-svn-ignore(){
   local msg="=== $FUNCNAME :"
   local tmp=/tmp/env/$FUNCNAME/ignore && mkdir -p $(dirname $tmp)    
   [ "$(tg-projdir)" != "$PWD"  ] && echo $msg ERROR not projdir && return 1  

   $FUNCNAME- > $tmp
   echo $msg setting svn:ignore property in PWD $PWD
   
   svn propset svn:ignore -F $tmp .
   svn pg svn:ignore .

   echo $msg with ignoring 
   svn status
   echo $msg no-ignore
   svn status --no-ignore
}


tg-svn-add-(){
  local msg="=== $FUNCNAME :"
  local dir=$1

  [ ! -d "$dir" ] && echo $msg no such dir && return 1

  echo $msg entering $dir 
  cd $(dirname $dir)

  [ -d "$dir/.svn" ] && svn revert $dir   
  svn -N add $dir
  cd $dir 
 
  local item 
  for item in $(ls -1 . | grep -v .svn) 
  do
     local act
     case $item in
       data) act=SKIP ;;
      *.ini) act=SKIP ;;
      *.pyc) act=SKIP ;;
          *) act=ADD ;;     
     esac
     local path=$dir/$item
     echo $msg $act $path

     if [ -d "$path" ]; then
        case $act in
           ADD) tg-svn-add- $path ;;
        esac
     else
       case $act in
          ADD) svn add $path  ;;
       esac
     fi
  done 
}


tg-svn-add(){

  cd $(tg-projdir)
  tg-svn-add- $PWD

  cd $(tg-projdir)
  tg-svn-ignore 

}




tg-ini(){ echo $(tg-projdir)/development.ini; }
tg-setup(){ paster setup-app $(tg-ini) ; }
tg-serve(){ paster serve $(tg-ini) ; }

tg-conf(){ $FUNCNAME- | python ; }
tg-conf-(){ cat << EOC
from configobj import ConfigObj
c = ConfigObj( "$(tg-ini)" , interpolation=False )
c['app:main']['sqlalchemy.url'] = "$(private-val DATABASE_URL)"
c['DEFAULT']['debug'] = "false"
c.write()
EOC
}



tg-scd(){   cd $(tg-srcdir) ; }
tg-cd(){    cd $(tg-projdir) ; }


