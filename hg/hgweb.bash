# === func-gen- : hg/hgweb fgp hg/hgweb.bash fgn hgweb fgh hg
hgweb-src(){      echo hg/hgweb.bash ; }
hgweb-source(){   echo ${BASH_SOURCE:-$(env-home)/$(hgweb-src)} ; }
hgweb-vi(){       vi $(hgweb-source) ; }
hgweb-env(){      elocal- ; }
hgweb-usage(){  cat << EOU

HGWEB
=======

List of solutions for remote Mercurial access
-----------------------------------------------

* http://mercurial.selenic.com/wiki/PublishingRepositories

HGWeb is only one way of providing remote access to Mercurial repos. 
The simplicity of just using SSH is attractive, see hgssh- for details.

HGWeb
-------

Deployments:

* C  supervisor controlled SCGI + apache modscgi 
* N  apache modwsgi embedded


* http://mercurial.selenic.com/wiki/HgWebDirStepByStep


     hgweb-hgrc-
         demo to stdout the hook that needs to be added 
         to .hg/hgrc to enable auto updates of the web interface for
         new repos
             http://mercurial.selenic.com/wiki/modwsgi


Mercurial clones in : /var/hg/repos 

Will show up in the list   
          http://belle7.nuu.edu.tw/hg/

with URLs like :
          http://belle7.nuu.edu.tw/hg/AuthKit/

hgweb-vhgpy 
         create a virtual python environment to house mercurial 



EOU
}
hgweb-cd(){  cd $(hgweb-dir); }

hgweb-name(){     echo hg ; }
hgweb-dir(){      echo /var/hg ; }
hgweb-confpath(){ echo $(hgweb-dir)/$(hgweb-name).ini ; }  
hgweb-wsgipath(){ echo $(apache- ; apache-cgidir)/$(hgweb-name).wsgi ; }
hgweb-scgipath(){ echo $(apache- ; apache-cgidir)/$(hgweb-name).scgi ; }
hgweb-edit(){     sudo vi $(hgweb-confpath) ; }

hgweb-build(){
   local msg="=== $FUNCNAME :" 
   [ -z "$VIRTUAL_ENV" ] && echo $msg ERROR you must be inside a virtual env first && return 1
   hgweb-prep
   hgweb-conf

   hgweb-wsgi
   hgweb-scgi

   hgweb-modwsgi-apache
   hgweb-selinux
}

hgweb-vhgdir(){      echo $(local-base)/env/vhg ; }
hgweb-vhg(){         . $(hgweb-vhgdir)/bin/activate  ; }
hgweb-vhgwipe(){
  rm -rf $(hgweb-vhgdir)
}
hgweb-vhgcreate(){
  local msg="=== $FUNCNAME :"

  local v=$(python-v)  ## only port installed stuff has the -2.5
  local py=$(which python)
  local ans
  read -p "$msg create virtual python with BASELINE $py , enter YES to proceed " ans
  [ "$ans" != "YES" ] && echo $msg skipped && return 1
  [ "$(which virtualenv)" == "" ] && echo $msg ERROR no virtualenv && return 1
  [ -n "$VIRTUAL_ENV" ] && echo $msg ERROR you must NOT be indside one to create one && return 1

  virtualenv $(hgweb-vhgdir)
  hgweb-vhg
  which python
  which easy_install$v

  easy_install$v mercurial
  easy_install$v ipython
  easy_install$v flup
  easy_install$v scgi

  hgweb-vhgselinux
  deactivate
}


hgweb-vhgselinux(){
  apache-
  python-
  apache-chcon $(hgweb-vhgdir)/lib/python$(python-major)/

  ## for SCGI to work had to enable this in order to avoid "name_connect" denial : 
  sudo setsebool httpd_can_network_connect 1 

}


hgweb-prep(){
  local msg="=== $FUNCNAME :"

  apache-
  local dirs="repos backup"
  local rdir
  for rdir in $dirs ; do
     local dir=$(hgweb-dir)/$rdir
     local cmd="sudo mkdir -p $dir"
     echo $msg $cmd
     eval $cmd
     cmd="sudo chown $USER $dir" 
     echo $msg $cmd
     eval $cmd

     apache-chown $dir
     apache-chcon $dir
  done 

}

hgweb-hgrc-(){ cat << EOC
## $FUNCNAME add to .hg/hgrc of repo for auto-reloading   
[hooks]
changegroup =
# reload wsgi application
changegroup.mod_wsgi = touch $(hgweb-wsgipath)
EOC
}

hgweb-conf-(){ cat << EOC
#[web]
#style = gitweb
[paths]
/ = $(hgweb-dir)/repos/**
/backup = $(hgweb-dir)/backup/**
EOC
}
hgweb-conf(){
  local msg="=== $FUNCNAME :"
  local conf=$(hgweb-confpath)
  local tmp=/tmp/env/$FUNCNAME/$(basename $conf) && mkdir -p $(dirname $tmp)
  echo $msg $conf writing to $tmp
  $FUNCNAME-
  $FUNCNAME- > $tmp
  local cmd="sudo cp $tmp $conf"
  echo $msg \"$cmd\"
  eval $cmd
}


hgweb-wsgi-(){ cat << EOC
ALLDIRS = ["$VIRTUAL_ENV/lib/python$(python-;python-major)/site-packages"]
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

from mercurial.hgweb.hgweb_mod import hgweb
from mercurial.hgweb.hgwebdir_mod import hgwebdir
application = hgwebdir('$(hgweb-confpath)')
EOC
}
hgweb-wsgi(){
  local msg="=== $FUNCNAME :"
  local wsgi=$(hgweb-wsgipath)
  local tmp=/tmp/env/$FUNCNAME/$(basename $wsgi) && mkdir -p $(dirname $tmp)
  echo $msg $wsgi writing to $tmp
  $FUNCNAME-
  $FUNCNAME- > $tmp
  local cmd="sudo cp $tmp $wsgi"
  echo $msg \"$cmd\"
  eval $cmd
}

hgweb-modwsgi-apache-(){ 
  local path=$(hgweb-wsgipath)
cat << EOC
## $FUNCNAME
WSGIScriptAlias /$(hgweb-name) $path
<Directory $(dirname $path)
    Order deny,allow
    Allow from all
</Directory> 
EOC
}
hgweb-modwsgi-apache(){
  local msg="=== $FUNCNAME :"
  $FUNCNAME- 
  echo $msg use \"apache-edit\" to incorporate the above 
}




hgweb-selinux(){
  apache-
  apache-chcon $(hgweb-dir)
}


hgweb-scgi-(){ 
   modscgi-
   cat << EOC

## $FUNCNAME in collaboration with modscgi-
## http://trac.saddi.com/flup/wiki/FlupServers
## http://einsteinmg.dyndns.org/projects/mercurial/hgwebdir1_fcgi  
## where to config the port 

from mercurial.hgweb.hgweb_mod import hgweb
from mercurial.hgweb.hgwebdir_mod import hgwebdir
from mercurial.hgweb.request import wsgiapplication

def app_maker():
    return hgwebdir('$(hgweb-confpath)')

from flup.server.scgi import WSGIServer
WSGIServer(wsgiapplication(app_maker), bindAddress=("127.0.0.1", $(local-port hg) ) ).run()

EOC
}

hgweb-scgi(){
  local msg="=== $FUNCNAME :"
  local scgi=$(hgweb-scgipath)
  local tmp=/tmp/env/$FUNCNAME/$(basename $scgi) && mkdir -p $(dirname $tmp)
  echo $msg $scgi writing to $tmp
  $FUNCNAME-
  $FUNCNAME- > $tmp
  local cmd="sudo cp $tmp $scgi"
  echo $msg \"$cmd\"
  eval $cmd
}


hgweb-scgi-run(){
  cd /tmp
  hgweb-vhg
  which python 
  local cmd="python $(hgweb-scgipath)"
  echo $msg $cmd
  eval $cmd
}


## supervisor hookup 

hgweb-sv(){  sv-;sv-add $FUNCNAME- hgweb.ini ; }
hgweb-sv-(){ 
   hgweb-vhg
   cat << EOC
[program:hgweb]
command=$(which python) $(hgweb-scgipath)
redirect_stderr=true
autostart=true
EOC
}


