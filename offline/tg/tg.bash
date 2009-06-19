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
   tg-eggcache
   tg-wsgi-deploy
}


tg-eggcache-dir(){
   case ${USER:-nobody} in 
      nobody|apache|www) echo /var/cache/tg ;;
                      *) echo $HOME ;;
    esac
}

tg-eggcache(){
   local cache=$(tg-eggcache-dir)
   [ "$cache" == "$HOME" ] && echo $msg cache is HOME:$HOME skipping && return 0

   echo $msg createing egg cache dir $cache
   sudo mkdir -p $cache
   apache-
   apache-chown $cache
   sudo chcon -R -t httpd_sys_script_rw_t $cache
   ls -alZ $cache
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

# sys.path.insert( 0, "$(tg-projdir)" )
# sys.path.insert( 1, "$(tg-srcdir)/lib/python$(python-major)" )

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


tg-hmac-kludge(){
   local msg="=== $FUNCNAME :"
   local target=${1:-N}
   [ "$NODE_TAG" != "G" ] && echo $msg this must be done from G && return 1 

   scp /opt/local/lib/python2.5/hmac.py $target:$(tg-srcdir $target)/lib/python2.4/
}


tg-hmac-test(){ $FUNCNAME- | python ; }
tg-hmac-test-(){ cat << EOT
import pkg_resources
pkg_resources.get_distribution('Beaker').version
import beaker.session
print beaker.session.sha1 
import hmac
print hmac.new('test', 'test', beaker.session.sha1).hexdigest()
print hmac.__file__
EOT
}




tg-srcfold(){  echo $(local-base $*)/env ; }
tg-srcnam(){   echo tg2env ; }
tg-srcdir(){   echo $(tg-srcfold $*)/$(tg-srcnam) ; }
tg-projname(){ echo OfflineDB ; }
tg-projdir(){  echo $(tg-dir)/$(tg-projname) ; }
tg-quickstart(){

   cd $(tg-dir)
   #  currently cannot install hashlib into py2.4 on N ... so skip the auth, see #205
   paster quickstart --auth --noinput $(tg-projname)
   #paster quickstart --noinput $(tg-projname)

   cd $(tg-projdir)
   python setup.py develop --uninstall
   python setup.py develop

   ## customize the ini with DATABASE_URL + ...
   tg-conf   

   ## labelling and ownership
   tg-selinux
   tg-datadir

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


