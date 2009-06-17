tg-src(){      echo offline/tg/tg.bash ; }
tg-source(){   echo ${BASH_SOURCE:-$(env-home)/$(tg-src)} ; }
tg-dir(){      echo $(dirname $(tg-source)) ; }
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
}

tg-activate(){ 
  cd $(tg-srcdir) 
  . bin/activate  
}

tg-install(){
   local dir=$(tg-srcdir)
   local fld=$(dirname $dir)
   local nam=$(basename $dir)
   mkdir -p $fld && cd $fld
   virtualenv --no-site-packages $nam
   tg-activate
   easy_install -i http://www.turbogears.org/2.0/downloads/current/index tg.devtools
}

tg-srcfold(){ echo $(local-base)/env ; }
tg-mode(){ echo bootstrap ; }
tg-srcnam(){ 
   case ${1:-$(tg-mode)} in
     bootstrap) echo tg2env ;;
   esac
}
tg-srcdir(){  echo $(tg-srcfold)/$(tg-srcnam) ; }

tg-projname(){ echo OfflineDB ; }
tg-projdir(){ echo $(tg-dir)/$(tg-projname) ; }
tg-quickstart(){

   cd $(tg-dir)
   #  currently cannot install hashlib into py2.4 on N ... so skip the auth, see #205
   #paster quickstart --auth --noinput $(tg-projname)
   paster quickstart --noinput $(tg-projname)

   cd $(tg-projdir)
   python setup.py develop

   tg-conf

}

tg-ini(){ echo $(tg-projdir)/development.ini; }
tg-setup(){ paster setup-app $(tg-ini) ; }
tg-serve(){ paster serve $(tg-ini) ; }

tg-conf(){ $FUNCNAME- | python ; }
tg-conf-(){ cat << EOC
from configobj import ConfigObj
c = ConfigObj( "$(tg-ini)" , interpolation=False )
c['app:main']['sqlalchemy.url'] = "$(private-val DATABASE_URL)"
c.write()
EOC
}



tg-scd(){   cd $(tg-srcdir) ; }
tg-cd(){    cd $(tg-projdir) ; }


