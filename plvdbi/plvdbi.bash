# === func-gen- : plvdbi/plvdbi fgp plvdbi/plvdbi.bash fgn plvdbi fgh plvdbi
plvdbi-src(){      echo plvdbi/plvdbi.bash ; }
plvdbi-source(){   echo ${BASH_SOURCE:-$(env-home)/$(plvdbi-src)} ; }
plvdbi-vi(){       vi $(plvdbi-source) ; }

plvdbi-env(){      
   elocal- ; 
   export PL_PROJNAME=plvdbi
   export PL_PROJDIR=$(plvdbi-dir)
   export PL_CONFNAME=production
   export PL_OPTS=" --server-name scgi_thread "
   export PL_VIP=dbi    #$(rum-;rum-dir)
   #export PL_CONFNAME=development
   pl-
}

plvdbi--(){
  vip-
  vip-activate $(pl-vip)
}

plvdbi-usage(){
  cat << EOU

    NB you must activate the approriate python virtual environment
    before these commands will work, eg with "rum-"


     Basis vars : 
       PL_PROJNAME : $PL_PROJNAME
       PL_PROJDIR  : $PL_PROJDIR
       PL_CONFNAME : $PL_CONFNAME
       PL_OPTS     : $PL_OPTS
       PL_VIP      : $PL_VIP
   
     Derived
       pl-confpath : $(pl-confpath)

     plvdbi-src : $(plvdbi-src)

     plvdbi-preqs 
         check the system python (or basis python) pre-requisites to the 
         installation and running namely :

               setuptools 
               virtualenv
               pip
               MySQLdb  + 

     plvdbi-vinstall 
         the installation will create a virtual python environment and 
         install a large number of required packages into this

     plvdbi-statics
         collect statics (.css, .js, etc...) from the source distributions
         and place them where the web server delivers them from 

         NB : javascript issues such as pages failing to fully load 
              are most likely caused by the statics becoming out of step with the sources 

     plvdbi-serve   
        interactive server run ... visible at http://localhost/dbi/ 
        after apache or other webserver is configured 

     plvdbi-versions
        report the versions of the more important packages : $(plvdbi-pkgs)


     plvdbi-curl/urllib/grab
          Usage :  
             plvdbi-urllib "SimPmtSpecDbis.json?limit=10&offset=50"
             plvdbi-curl   "SimPmtSpecDbis.json?limit=1&offset=100"
             plvdbi-grab   "SimPmtSpecDbis.json?limit=2&offset=100"
                   demo json access


     ###############  for development only ######################
 
     plvdbi-make-config
        create deployment config file from template :
             plvdbi/plvdbi/config/deployment.ini_tmpl

     plvdbi-shell
          
          gives error ... invalid literal for int() arising from :
          the response of /_test_vars not being covertible to integer 
             # Query the test app to setup the environment
             tresponse = test_app.get('/_test_vars')
             request_id = int(tresponse.body)

     plvdbi-archive-tw-resources
           collect the statics for deployment/serving from web server (apache/nginx/lighttpd) rather than webapp
               http://toscawidgets.org/documentation/ToscaWidgets/deploy.html
               http://projects.roggisch.de/tw/aggregation.html

      plvdbi-freeze
          freeze the state of python into $(pl-pippath)

      plvdbi-thaw
           install based on the versions/repos/clones specified in $(pl-pippath)
           for example ... into a test virtual python :
                 PL_VPYDIR=$(local-base)/env/vrum plvdbi-thaw


EOU
}

plvdbi-dir(){     echo $(env-home)/plvdbi ; }
plvdbi-cd(){      cd $(plvdbi-dir); }
plvdbi-mate(){    mate $(plvdbi-dir) ; }
plvdbi-workdir(){ echo /tmp/env/plvdbi/workdir ; }

plvdbi-build(){

    local msg="=== $FUNCNAME :"
    vdbi-
    vdbi-build
    [ ! $? -eq 0 ] && echo $msg ABORT after vdbi-build      && return 1  || echo $msg vdbi-build OK

    pl-
    pl-build 
    [ ! $? -eq 0 ] && echo $msg ABORT after pl-build        && return 1  || echo $msg pl-build OK

    authkit-
    authkit-build  
    [ ! $? -eq 0 ] && echo $msg ABORT after authkit-build   && return 1  || echo $msg authkit-build OK

    plvdbi-install 
    [ ! $? -eq 0 ] && echo $msg ABORT after -install && return 1  || echo $msg -install OK

    pldep-
    pldep-selinux  
    [ ! $? -eq 0 ] && echo $msg ABORT after -selinux && return 1  || echo $msg -selinux OK

    plvdbi-make-config 
    [ ! $? -eq 0 ] && return 1

    plvdbi-statics
    [ ! $? -eq 0 ] && return 1
}


plvdbi-statics(){

    plvdbi-statics-check
    [ ! $? -eq 0 ] && return 1

    plvdbi-archive-tw-resources
    [ ! $? -eq 0 ] && return 1

    plvdbi-statics-selinux  
    [ ! $? -eq 0 ] && return 1
}

plvdbi-develop(){ pl-setup develop ; }


plvdbi-serve(){
  local msg="=== $FUNCNAME :" 

  plvdbi-private-check
  [ ! "$?" == "0" ] && echo $msg ABORT -private-check fails && return 1

  plvdbi--
  local iwd=$PWD 
  local dir=$(plvdbi-workdir)
  mkdir -p $dir && cd $dir
  pl-serve 
  cd $iwd
}


plvdbi-sv(){
  ## customized via the coordinate envvars
  
  plvdbi-private-check
  plvdbi--
  pl-sv
}

plvdbi-port(){
   private-
   private-val PLVDBI_PORT
}

plvdbi-webopen-ip(){
   local tag=${1:-G}
   iptables-
   IPTABLES_PORT=$(plvdbi-port) iptables-webopen-ip $(local-tag2ip $tag)
}


plvdbi-private-check(){
   private-
   local msg="=== $FUNCNAME :"
   local pport=$(private-val PLVDBI_PORT) 
   local lport=$(local-port plvdbi)
   [ "$pport" != "$lport" ] && echo $msg ABORT port mismatch pport $pport lport $lport && return 1 
   return 0
}

plvdbi-make-config-(){
   private-
   cat << EOC
           email_to=$(private-val PLVDBI_EMAIL_TO) 
        smtp_server=$(private-val PLVDBI_SMTP_SERVER) 
               port=$(private-val PLVDBI_PORT) 
   error_email_from=$(private-val PLVDBI_ERROR_EMAIL_FROM)
EOC
}

plvdbi-make-config(){
   local msg="=== $FUNCNAME :"
   [ "$(pl-confname)" == "development" ] && echo $msg ABORT this is not applicable to the developmemnt.ini ... used for production only && return 1
   plvdbi-private-check
   [ ! "$?" -eq "0" ] && echo $msg ABORT -private-check failed &&  return 1

   local ini=$(pl-confpath)
   local cmd="paster make-config plvdbi $ini $(echo $(plvdbi-make-config-)) ; svn revert $ini "
   echo $msg \"$cmd\"
   eval $cmd
}

plvdbi-shell(){
   local tmp=/tmp/env/$FUNCNAME && mkdir -p $tmp
   local iwd=$PWD
   cd $tmp
   pl-shell
   cd $iwd
}

plvdbi-statics-dir(){  echo $(plvdbi-dir)/plvdbi/public/toscawidgets ; }
plvdbi-archive-tw-resources(){
   local msg="=== $FUNCNAME :"
   cd $(plvdbi-dir)
   [ ! -f "setup.cfg" ] && echo $msg ABORT need setup.cfg to define the distributions to get resources from && return 1
   cat setup.cfg
   plvdbi--
   local cmd="python setup.py archive_tw_resources -f --output $(plvdbi-statics-dir)"
   echo $msg \"$cmd\"
   eval $cmd
}

plvdbi-statics-check(){
   private-
   local psd=$(private-val PLVDBI_STATICS_DIR)
   [ "$psd" != "$(plvdbi-statics-dir)" ] && echo $msg ERROR inconsistency between private-val PLVDBI_STATICS_DIR and the function && return 1
   return 0
}

plvdbi-statics-selinux(){
   apache-chcon $(plvdbi-statics-dir)
}

plvdbi-statics-apache-(){  cat << EOC
Alias /dbi/toscawidgets/ $(plvdbi-statics-dir)/ 
<Directory $(plvdbi-statics-dir)>
Order deny,allow
Allow from all
</Directory>
EOC
}

plvdbi-statics-apache(){
  local msg="=== $FUNCNAME :"
  $FUNCNAME- 
  echo $msg incoporate smth like the above with apache-edit 
}


plvdbi-preqs(){
  local msg="=== $FUNCNAME :"
  [ "$(which hg)" == "" ] && echo $msg FAILURE ... Mercurial is missing && return 1
  python -c "import MySQLdb "
  vip-
  vip-preqs
  [ ! "$?" == "0" ] && echo $msg FAILURE ... && return 1  
  return 0
}



## the below vip installs can be contracted to a single line pip install once stabilization is achieved, split for faster development 

plvdbi-req(){ vi $(pl-pippath) ;  }
plvdbi-basis(){     vip- ; $FUNCNAME- | vip-install $(pl-vip) ; }
plvdbi-basis-(){    cat $(pl-pippath) ; }

plvdbi-editables(){ vip- ; $FUNCNAME- | vip-install $(pl-vip) ; }
plvdbi-editables-(){  cat << EOE
-e hg+http://bitbucket.org/bbangert/pylons/@ccd78f4b1f3c2378b6ecc325c17bd0a7fca9d5bb#egg=Pylons-tip
-e hg+http://toscawidgets.org/hg/ToscaWidgets@78787813b0e33065de53a61d03399fda438940bd#egg=ToscaWidgets-0.9.8dev_20091019-py2.4-dev
-e hg+http://toscawidgets.org/hg/tw.jquery@e0e598dad42aa3234aaf05093e470ea94a3e2581#egg=tw.jquery-tip
-e hg+http://hg.python-rum.org/tw.rum@a521028224f50b4d7a37d7098cffb218f1fa4d2b#egg=tw.rum-tip

-e hg+http://belle7.nuu.edu.tw/hg/AuthKitPy24#egg=AuthKitPy24-tip
-e hg+http://belle7.nuu.edu.tw/hg/rum#egg=rum-tip
-e svn+http://dayabay.phys.ntu.edu.tw/repos/env/trunk/private#egg=private-dev  
EOE
}

plvdbi-qeditables(){ vip- ;  $FUNCNAME- | vip-install $(pl-vip) ; }
plvdbi-qeditables-(){  cat << EOE
-e ../src/pylons-tip
-e ../src/toscawidgets 
-e ../src/tw.jquery-tip
-e ../src/rum-tip
-e ../src/tw.rum-tip

-e ../src/authkitpy24-tip
-e ../src/private  
EOE
}

plvdbi-primes(){ vip- ;  $FUNCNAME- | vip-install $(pl-vip) ; }
plvdbi-primes-(){  cat << EOE
-e $(env-home)/vdbi
-e $(env-home)/plvdbi
EOE
}


plvdbi-vinstall(){
  local msg="=== $FUNCNAME :"
  plvdbi-preqs
  [ ! "$?" == "0" ] && echo $msg install system python pre-requisites then try again && return 1
  plvdbi-basis
  [ ! "$?" == "0" ] && echo $msg plvdbi-vip failed && return 1 
  plvdbi-editables
  [ ! "$?" == "0" ] && echo $msg plvdbi-editables failed && return 1 
  plvdbi-qeditables
  [ ! "$?" == "0" ] && echo $msg plvdbi-qeditables failed && return 1 
  plvdbi-primes
  [ ! "$?" == "0" ] && echo $msg plvdbi-primes failed && return 1 



}


plvdbi-pkgs(){      private- ; private-val VDBI_PACKAGES ; }
plvdbi-versions(){  $FUNCNAME- | $(vip-dir $(pl-vip))/bin/python ; }
plvdbi-versions-(){ cat << EOV
import pkg_resources as pr
for pkg in '$(plvdbi-pkgs)'.split(','):
    print pr.get_distribution(pkg)
EOV
}


plvdbi-url(){ echo http://$(plvdbi-node)/dbi/${1:-SimPmtSpecDbis.json} ;  }
plvdbi-node(){ echo localhost ; }

plvdbi-curl-(){ cat << EOC
curl -v -d username=$(private-val DAYABAY_USER) -d password=$(private-val DAYABAY_PASS) "$(plvdbi-url $*)"
EOC
}

plvdbi-curl(){
  private-
  eval $($FUNCNAME- $*)
}

plvdbi-urllib(){ $FUNCNAME- $* | python ; }
plvdbi-urllib-(){ private- ; cat << EOC
import urllib2
import urllib
creds = urllib.urlencode({  'username':"$(private-val DAYABAY_USER)", 'password':"$(private-val DAYABAY_PASS)", } )
req = urllib2.Request('$(plvdbi-url $*)', creds)
print urllib2.urlopen(req).read()
EOC
}

plvdbi-grab(){ python $(plvdbi-dir)/grab.py "$(plvdbi-url $*)" ; }


