# === func-gen- : vdbi/vdbi fgp vdbi/vdbi.bash fgn vdbi fgh vdbi
vdbi-src(){      echo vdbi/vdbi.bash ; }
vdbi-source(){   echo ${BASH_SOURCE:-$(env-home)/$(vdbi-src)} ; }
vdbi-vi(){       vi $(vdbi-source) ; }
vdbi-env(){      elocal- ; }
vdbi-usage(){
  cat << EOU
     vdbi-src : $(vdbi-src)
     vdbi-dir : $(vdbi-dir)

     vdbi-build
         installs dependencies, Rum, RumAlchemy,tw.rum

     vdbi-extras
        get and install ToscaWidgets + tw.jquery from mercurial repo
        they dont have recent enough pypi releases to fit into the setup.py install


EOU
}
vdbi-dir(){ echo $(env-home)/vdbi ; }
vdbi-cd(){  cd $(vdbi-dir); }
vdbi-mate(){ mate $(vdbi-dir) ; }

vdbi-ipy(){
   local msg="=== $FUNCNAME :"
   rum-
   cd /tmp
   echo $msg activate rum env and start ipython from $PWD ... enter \"vdbi\" into ipython for debug running 
   ipython 
}



vdbi-build(){
  local msg="=== $FUNCNAME :"
  echo $msg for a pip controlled build use : plvdbi-vinstall 
}


vdbi-build-uncontrolled(){
  local msg="=== $FUNCNAME :"
  #rum-
  #rum-build

  vip-
  local name=${1:-dbi}
  local vdir=$(vip-dir $name)

  vip-preqs               ## check the triplet : setuptools/virtualenv/pip are installed
  [ ! $? -eq 0 ] && return 1
  vip-create $name        ## create virtual environment if not existing
  [ ! $? -eq 0 ] && return 1
  vip-activate $name      ## get into the virtual env
  [ ! $? -eq 0 ] && return 1

  ## check are in the virtual env 
  [ "$(which python)" != "$vdir/bin/python" ]  && echo $msg ABORT must be inside virtual python env to proceed && return 1

  vdbi-install
  [ "$?" != "0" ] && echo $msg abort aftert vdbi-install &&  return 1
  vdbi-extras
  [ "$?" != "0" ] && echo $msg abort aftert vdbi-extras &&  return 1
  vdbi-selinux 
  [ "$?" != "0" ] && echo $msg abort aftert vdbi-selinux &&  return 1
  vdbi-users-path 
  [ "$?" != "0" ] && echo $msg abort aftert vdbi-users-path &&  return 1
  vdbi-logdir    
  [ "$?" != "0" ] && echo $msg abort aftert vdbi-logdir &&  return 1
  return 0
}

vdbi-setup(){
   local msg="=== $FUNCNAME :"
   vdbi-cd
   local cmd="python setup.py $*"
   echo $msg $cmd ... from $PWD with $(which python)
   eval $cmd
}

vdbi-install(){ vdbi-setup develop ; }

vdbi-extras(){
  twdev-
  twdev-build
}

vdbi-selinux(){
   apache-
   apache-chcon $(vdbi-dir)
}

vdbi-users-path(){
  local msg="=== $FUNCNAME :"
  echo $msg
  private-
  local vap=$(private-get VDBI_USERS_PATH)
  [ "$vap" == "" ] && echo $msg must have a private key VDBI_USERS_PATH in $(apache-private-path) that points to the vdbi users file : user private-edit to set it && return 1
  [ ! -f "$vap" ] && echo $msg there is no users file at $vap : you need to create this to enable login  && return 1
  apache-
  apache-chcon $vap
  apache-chown $vap
  return 0
}

vdbi-logdir(){
  local msg="=== $FUNCNAME :"
  echo $msg
  private-
  local logdir=$(private-get VDBI_LOGDIR)
  [ "$logdir" == "" ] && echo $msg must have a private key VDBI_LOGDIR in $(apache-private-path)  : user private-edit to set it && return 1

  mkdir -p $logdir
  apache-chown $logdir -R
  apache-chcon $logdir 

}

vdbi-check(){
   curl --user-agent MSIE http://127.0.0.1:6060
}

