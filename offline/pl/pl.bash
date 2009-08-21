# === func-gen- : offline/pl/pl.bash fgp offline/pl/pl.bash fgn pl
pl-src(){      echo offline/pl/pl.bash ; }
pl-source(){   echo ${BASH_SOURCE:-$(env-home)/$(pl-src)} ; }
pl-dir(){      echo $(env-home $*)/$(dirname $(pl-src)) ; }
pl-vi(){       vi $(pl-source) ; }
pl-env(){      elocal- ; }
pl-usage(){
  cat << EOU
     pl-src : $(pl-src)

     pl-install
           set up virtual env and populate with Pylons

     pl-quickstart 
           set up project 


     pl-deploy 


=== pl-wsgi : writing /tmp/env/pl-wsgi/helloworld.wsgi
sudo cp -f /tmp/env/pl-wsgi/helloworld.wsgi /var/www/cgi-bin/helloworld.wsgi
=== modwsgi-app-conf : "WSGIScriptAlias /helloworld /var/www/cgi-bin/helloworld.wsgi" needs to be added to apache-conf
WSGIPythonHome /data1/env/local/env/tg2env
=== modwsgi-virtualenv : "WSGIPythonHome /data1/env/local/env/tg2env" already hooked up to apache

       hmmm : multiple virtualenv in single apache ?


EOU
}


pl-preq-install-yum(){  [ "$(which hg)" == "" ] && sudo yum  install mercurial ; } 
pl-preq-install-port(){ [ "$(which hg)" == "" ] && sudo port install mercurial ; }
pl-preq-install(){
   pkgr-
   case $(pkgr-cmd) in 
      yum) $FUNCNAME-yum ;;
     port) $FUNCNAME-port ;;
   esac
}

pl-preq(){
    local msg="=== $FUNCNAME :"
    echo $msg preqs for the baseline python ... not the virtualized one
    python-
    [ "$(python-version)"     != "2.4.3" ]  && echo $msg untested python version

    virtualenv-
    virtualenv-get
    [ "$(virtualenv-version)" != "1.3.3" ] && echo $msg untested virtualenv  
}

pl-srcfold(){  echo $(local-base $*)/env ; }
pl-srcnam(){   echo plenv ; }
pl-srcdir(){   echo $(pl-srcfold $*)/$(pl-srcnam) ; }
#pl-projname(){ echo OfflineDB ; }
pl-projname(){ echo helloworld ; }
pl-projdir(){  echo $(pl-dir)/$(pl-projname) ; }
pl-cd(){       cd $(pl-projdir) ; }   

pl-activate(){
   local act=$(pl-srcdir)/bin/activate
   [ -f "$act" ] && . $act
 }


pl-install(){

   local msg="=== $FUNCNAME :"
  
   pl-preq-install
   pl-preq
  
   local dir=$(pl-srcdir)
   local fld=$(dirname $dir)
   local nam=$(basename $dir)
   mkdir -p $fld && cd $fld
   virtualenv --no-site-packages $nam
   pl-activate
   cd $(pl-srcdir)

   hg clone https://www.knowledgetap.com/hg/pylons-dev Pylons

   cd Pylons
   python setup.py develop 

   pl-chcon
}

pl-chcon(){
   modwsgi-
   modwsgi-baseline-chcon $(pl-srcdir)
}


  
pl-quickstart(){
  pl-activate
  cd $(pl-dir)  
  local proj=$(pl-projname)
  [ -d "$proj" ] && echo $msg ERROR proj $proj exists already && return 1 

  paster create -t pylons $(pl-projname) 
  cd $(pl-projname)

  ## this with get the dependencies, such as SQLAlchemy 
  python setup.py develop

  ## could be handled by adding requirements to the setup.py of the proj 
  easy_install configobj
  easy_install ipython  
  easy_install MySQL-python

  pl-conf
  # python-ln $(env-home) env   ## for env.base.private.Private access
}


pl-conf(){ $FUNCNAME- | python ; }
pl-conf-(){ cat << EOC
# this could be done be either the base python or the virtualized one 
from configobj import ConfigObj
c = ConfigObj( "$(pl-ini)" , interpolation=False )
c['app:main']['sqlalchemy.url'] = "$(private-val DATABASE_URL)"
c['DEFAULT']['debug'] = "false"
c.write()
EOC
}




pl-ini(){ echo $(pl-projdir)/development.ini ; }

pl-serve(){
  pl-activate
  cd $(pl-projdir) 
  paster serve --reload $(pl-ini)
}



pl-wsgi-(){  cat << EOS

#  $FUNCNAME 
#     http://code.google.com/p/modwsgi/wiki/VirtualEnvironments

ALLDIRS = ["$(pl-srcdir)/lib/python$(python-major)/site-packages"]

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
os.environ['PYTHON_EGG_CACHE'] = '$(pl-eggcache-dir)'

from paste.deploy import loadapp
application = loadapp('config:$(pl-projdir)/development.ini')

EOS
}

pl-wsgi(){
  local msg="=== $FUNCNAME :"
  local tmpd=/tmp/env/$FUNCNAME && mkdir -p $tmpd
  local tmp=$tmpd/$(pl-projname).wsgi
  echo $msg writing $tmp
  $FUNCNAME- > $tmp
  modwsgi-
  modwsgi-deploy $tmp
}

pl-eggcache-dir(){ echo /var/cache/pl ; }

pl-deploy(){

  apache-
  apache-eggcache $(pl-eggcache-dir)

  pl-wsgi
}




