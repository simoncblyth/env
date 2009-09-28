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
pl-srcnam(){   echo pldev ; }  
pl-srcdir(){   echo $(pl-srcfold $*)/$(pl-srcnam)/pylons ; }
pl-mate(){     mate $(pl-srcdir) ; }


pl-projname(){ echo ${PL_PROJNAME:-helloworld} ; }
pl-projdir(){  echo $(pl-dir)/$(pl-projname) ; }
pl-cd(){       cd $(pl-projdir) ; }   


pl-build(){

  local msg="=== $FUNCNAME :"
  [ -z "$VIRTUAL_ENV" ] && echo $msg ABORT are not inside virtualenv && return 1 
  [ "$(which python)" != "$VIRTUAL_ENV/bin/python" ] && echo  $msg ABORT wrong python && return 1


  pl-get
  pl-install
  pl-selinux 
  pl-eggcache
}




pl-get(){
   local msg="=== $FUNCNAME :"
   [ "$(which hg)" == "" ] && echo $msg no hg && return 1
   local dir=$(dirname $(pl-srcdir))
   local nam=$(basename $(pl-srcdir))

   mkdir -p $dir && cd $dir
   local cmd="hg clone http://bitbucket.org/bbangert/pylons/ $nam"
   echo $msg \"$cmd\" from $PWD
   eval $cmd 

   #hg clone https://www.knowledgetap.com/hg/pylons-dev Pylons
}

pl-install(){
   local msg="=== $FUNCNAME :"


   

   cd $(pl-srcdir)
   local cmd="python setup.py develop"
   echo $msg \"$cmd\"  ... from $PWD with $(which python)
   eval $cmd
}

pl-selinux(){
   apache-
   apache-chcon $(pl-srcdir)
}

pl-eggcache-dir(){ echo /var/cache/pl ; }
pl-eggcache(){
  apache-
  apache-eggcache $(pl-eggcache-dir)
}







pl-proj-deps(){
  pl-activate

  ## could be handled by adding requirements to the setup.py of the proj 
  easy_install configobj
  easy_install ipython  
  easy_install MySQL-python
}

  
pl-create(){
  pl-activate
  cd $(pl-dir)  
  local proj=${1:-$(pl-projname)}
  [ -d "$proj" ] && echo $msg ERROR proj $proj exists already && return 1 

  paster create -t pylons $proj 
  cd $proj

  ## this will get the dependencies, such as SQLAlchemy 
  python setup.py develop

  ## edit the development.ini adding DB coordinates etc..
  pl-conf

  # python-ln $(env-home) env   ## for env.base.private.Private access
}


pl-conf(){ 
   local msg="=== $FUNCNAME :"
   $FUNCNAME-
   [ ! -f "$(pl-ini)" ] && echo $msg ABORT no .ini file at $(pl-ini) && return 1
   $FUNCNAME- | python 
}
pl-conf-(){ 
    private-
    # this could be done be either the base python or the virtualized one 
    cat << EOC
#
#   C A U T I O N  :    D O    N O T   C O M M I T   T H E    I N I   :    $(pl-ini)
#
from configobj import ConfigObj
c = ConfigObj( "$(pl-ini)" , interpolation=False )
c['app:main']['sqlalchemy.url'] = "$(private-val DATABASE_URL)"
c['DEFAULT']['debug'] = "false"
c.write()
EOC
}




pl-ini(){ echo ${PL_INI:-$(pl-projdir)/development.ini} ; }

pl-serve(){
  local msg="=== $FUNCNAME :"
  pl-activate
  cd $(pl-projdir) 
  echo $msg serving $(pl-ini) from $PWD with $(which paster)
  paster serve --reload $(pl-ini)
}



pl-wsgi-(){  
  python-
  cat << EOS

#  $FUNCNAME 
#     http://code.google.com/p/modwsgi/wiki/VirtualEnvironments

ALLDIRS = ["$VIRTUAL_ENV/lib/python$(python-major)/site-packages"]

import sys
import site

sys.stdout = sys.stderr


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
os.environ['ENV_PRIVATE_PATH'] = '$(apache-private-path)'

from paste.deploy import loadapp
application = loadapp('config:$(pl-ini)')

EOS
}


pl-wsgi(){
  local msg="=== $FUNCNAME :"
  local tmpd=/tmp/env/$FUNCNAME && mkdir -p $tmpd
  local tmp=$tmpd/$(pl-projname).wsgi

  [ -z "$VIRTUAL_ENV" ] && echo $msg abort not in virtual env ... && return 1

  echo $msg writing $tmp ... using VIRTUAL_ENV $VIRTUAL_ENV
  $FUNCNAME- > $tmp
  modwsgi-
  modwsgi-deploy $tmp
}




pl-deploy(){

  pl-wsgi
}




## the below should be factored into sphinx- for minimal invokation from here

pl-book-dir(){  echo $(local-base $*)/env/PylonsBook ; }
pl-book-cd(){   cd $(pl-book-dir) ; }
pl-book-get(){
   local msg="=== $FUNCNAME :"
   local dir=$(dirname $(pl-book-dir))
   mkdir -p $dir && cd $dir
   local nam=$(basename $(pl-book-dir))
   [ ! -d "$nam" ] && hg clone https://hg.3aims.com/public/PylonsBook/ || echo $msg $nam is already cloned 
}
pl-book-builddir(){ echo .build ; }
pl-book-build(){
    pl-book-cd
    sphinx-build -b html . ./$(pl-book-builddir)
}
pl-book-open(){
   local name=${1:-index.html}
   open file://$(pl-book-dir)/$(pl-book-builddir)/$name
}
pl-book-build-latex(){
    pl-book-cd
    sphinx-build -b latex . ./$(pl-book-builddir)
}
pl-book-build-pdf(){
    pl-book-build-latex
    cd ./$(pl-book-builddir)
    pdflatex PylonsBook.tex
    pdflatex PylonsBook.tex
}
pl-book-open-pdf(){
   pl-book-open PylonsBook.pdf
}



